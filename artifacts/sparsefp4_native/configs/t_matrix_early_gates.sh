#!/usr/bin/env bash
# Recover the 100/250-step gate checkpoints: rerun the first 250 steps of
# each T-arm with identical seed/data/world-size (deterministic trajectory
# prefix) under the patched trainer that honors
# weight_only_checkpointing_steps. Writes checkpoint-50..250 next to the
# round-1 checkpoint-500 in the same output dirs.
set -u
REPO=/home/ec2-user/FastVideo
cd "$REPO"
source artifacts/sparsefp4_followup/configs/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
LOGDIR="$REPO/artifacts/sparsefp4_native/logs/t_matrix"
mkdir -p "$LOGDIR"
PROC=/mnt/nvme/scratch/sparsefp4_native/t_corpus_processed_t2v

train_arm() {  # $1 arm, $2 gpus, $3 pipeline, $4 lr, $5 port
  local ARM=$1 GPUS=$2 PIPELINE=$3 LR=$4 PORT=$5
  local NUM=$(echo "$GPUS" | tr ',' '\n' | wc -l)
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nnodes 1 --nproc_per_node "$NUM" --master_port "$PORT" \
    "fastvideo/training/$PIPELINE" \
    --num_gpus "$NUM" --sp_size 1 --tp_size 1 \
    --hsdp_replicate_dim "$NUM" --hsdp_shard_dim 1 \
    --model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --pretrained_model_name_or_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --data_path "$PROC/combined_parquet_dataset/" \
    --dataloader_num_workers 2 \
    --tracker_project_name "dqvsa_${ARM}_early" \
    --output_dir "checkpoints/dqvsa_$ARM" \
    --max_train_steps 250 \
    --train_batch_size 1 --train_sp_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_latent_t 20 --num_height 480 --num_width 832 --num_frames 77 \
    --enable_gradient_checkpointing_type "full" \
    --learning_rate "$LR" \
    --mixed_precision "bf16" \
    --weight_only_checkpointing_steps 50 \
    --training_state_checkpointing_steps 10000 \
    --weight_decay 0.01 --max_grad_norm 1.0 \
    --inference_mode False \
    --checkpoints_total_limit 15 \
    --training_cfg_rate 0.1 \
    --dit_precision "fp32" \
    --ema_start_step 0 --flow_shift 1 --seed 1000 \
    --VSA_decay_rate 0.9 --VSA_decay_interval_steps 1 --VSA_sparsity 0.9 \
    > "$LOGDIR/${ARM}_early.log" 2>&1
  echo "${ARM}_EARLY_RC=$?"
}

export FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_QAT_VSA256_ATTN
export FASTVIDEO_FA4_BWD_FALLBACK=1

( train_arm T1 0,1 wan_training_pipeline.py          1e-6 29661 ) &
( FASTVIDEO_DQVSA_NAIVE_BWD=1 \
  train_arm T2 2,3 wan_dqvsa_distillation_pipeline.py 1e-5 29662 ) &
( train_arm T3 4,5 wan_dqvsa_distillation_pipeline.py 1e-5 29663 ) &
wait
echo "T_MATRIX_EARLY_DONE"
