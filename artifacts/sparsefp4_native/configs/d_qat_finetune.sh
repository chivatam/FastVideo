#!/usr/bin/env bash
# Track D: QAT recovery fine-tune — SLA2-style fake-quant NVFP4 QK on the VSA
# fine branch (SPARSEFP4_QAT_VSA_ATTN), constant sparsity 0.9 (deployment
# point, no decay schedule), crush-smol latents, feasibility scale.
#   bash d_qat_finetune.sh <gpus e.g. "3"> <max_steps>
set -u
GPUS=${1:-3}
MAX_STEPS=${2:-400}
REPO=/home/ec2-user/FastVideo
cd "$REPO"
source artifacts/sparsefp4_followup/configs/env.sh

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_QAT_VSA_ATTN
export FASTVIDEO_FA4_BWD_FALLBACK=1
export CUDA_VISIBLE_DEVICES=$GPUS
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

torchrun --nnodes 1 --nproc_per_node "$NUM_GPUS" --master_port 29650 \
  fastvideo/training/wan_training_pipeline.py \
  --num_gpus "$NUM_GPUS" --sp_size 1 --tp_size 1 \
  --hsdp_replicate_dim "$NUM_GPUS" --hsdp_shard_dim 1 \
  --model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --pretrained_model_name_or_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --data_path data/crush-smol_processed_t2v/combined_parquet_dataset/ \
  --dataloader_num_workers 2 \
  --tracker_project_name sparsefp4_qat_recovery \
  --output_dir "checkpoints/sparsefp4_qat_recovery" \
  --max_train_steps "$MAX_STEPS" \
  --train_batch_size 1 \
  --train_sp_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_latent_t 20 \
  --num_height 480 --num_width 832 --num_frames 77 \
  --enable_gradient_checkpointing_type "full" \
  --learning_rate 1e-5 \
  --mixed_precision "bf16" \
  --weight_only_checkpointing_steps 200 \
  --training_state_checkpointing_steps 400 \
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
echo "QAT_TRAIN_RC=$?"
