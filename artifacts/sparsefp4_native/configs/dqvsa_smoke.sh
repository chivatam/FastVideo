#!/usr/bin/env bash
# DQ-VSA Stage-2 smoke test (<=20 steps): velocity distillation from frozen
# P4G-operator teacher into fake-quant NVFP4 student. Validates forward
# precision semantics, backward correctness, gradient finiteness, memory,
# checkpoint save/load, and native P4 serving of the trained weights.
#   bash dqvsa_smoke.sh <gpu> [max_steps]
set -u
GPU=${1:-0}
MAX_STEPS=${2:-20}
REPO=/home/ec2-user/FastVideo
cd "$REPO"
source artifacts/sparsefp4_followup/configs/env.sh

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_QAT_VSA256_ATTN
export FASTVIDEO_FA4_BWD_FALLBACK=1
export FASTVIDEO_DQVSA_GRAD_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU

OUT=checkpoints/dqvsa_smoke
rm -rf "$OUT"

torchrun --nnodes 1 --nproc_per_node 1 --master_port 29655 \
  fastvideo/training/wan_dqvsa_distillation_pipeline.py \
  --num_gpus 1 --sp_size 1 --tp_size 1 \
  --hsdp_replicate_dim 1 --hsdp_shard_dim 1 \
  --model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --pretrained_model_name_or_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --data_path data/crush-smol_processed_t2v/combined_parquet_dataset/ \
  --dataloader_num_workers 2 \
  --tracker_project_name dqvsa_smoke \
  --output_dir "$OUT" \
  --max_train_steps "$MAX_STEPS" \
  --train_batch_size 1 \
  --train_sp_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_latent_t 20 \
  --num_height 480 --num_width 832 --num_frames 77 \
  --enable_gradient_checkpointing_type "full" \
  --learning_rate 1e-5 \
  --mixed_precision "bf16" \
  --weight_only_checkpointing_steps "$MAX_STEPS" \
  --training_state_checkpointing_steps "$MAX_STEPS" \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --inference_mode False \
  --checkpoints_total_limit 2 \
  --training_cfg_rate 0.1 \
  --dit_precision "fp32" \
  --ema_start_step 0 \
  --flow_shift 1 \
  --seed 1000 \
  --VSA_decay_rate 0.9 \
  --VSA_decay_interval_steps 1 \
  --VSA_sparsity 0.9
RC=$?
echo "DQVSA_TRAIN_RC=$RC"
[ $RC -ne 0 ] && exit $RC

# ---- native P4 serving check with the trained checkpoint ----
SERVE=/mnt/nvme/scratch/sparsefp4_native/dqvsa_smoke_model
HF_SNAP=$(ls -d "$FV_SCRATCH"/hf-cache/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/*/ | head -1)
mkdir -p "$SERVE"
for c in model_index.json scheduler text_encoder tokenizer vae; do
  ln -sfn "$HF_SNAP/$c" "$SERVE/$c"
done
ln -sfn "$REPO/$OUT/checkpoint-$MAX_STEPS/transformer" "$SERVE/transformer"

"$FV_PYTHON" artifacts/sparsefp4_native/configs/p_run.py \
  --run-id dqvsa-smoke --arm P4 --prompt-index 0 --steps 10 \
  --model-path "$SERVE"
echo "DQVSA_SERVE_RC=$?"
