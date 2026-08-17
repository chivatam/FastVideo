#!/usr/bin/env bash
# DQ-VSA T-matrix: preprocess the generated corpus, then run T1/T2/T3
# (500 steps each, weight-only checkpoints every 50 steps -> 100/250/500
# gates) in parallel, 2 GPUs per arm.
#
#   T1  standard flow-matching QAT      (wan_training_pipeline,   LR 1e-6)
#   T2  velocity distillation, naive    (DQVSA + NAIVE_BWD=1,     LR 1e-5)
#       fake-quant-fwd / BF16-bwd
#   T3  velocity distillation, Attn-QAT (DQVSA default backward,  LR 1e-5)
#       consistent backward
#
# T0 = untrained P4 (existing pq-s090 videos). Same data/steps/batch for all.
set -u
REPO=/home/ec2-user/FastVideo
cd "$REPO"
source artifacts/sparsefp4_followup/configs/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
LOGDIR="$REPO/artifacts/sparsefp4_native/logs/t_matrix"
mkdir -p "$LOGDIR"

CORPUS=/mnt/nvme/scratch/sparsefp4_native/t_corpus
PROC=/mnt/nvme/scratch/sparsefp4_native/t_corpus_processed_t2v

# ---- preprocess (skip if done): v1_preprocess only supports 1 GPU, so
# split the manifest into 8 shards, preprocess in parallel, and combine
# (the parquet dataset walks the tree recursively).
if [ ! -d "$PROC/combined_parquet_dataset" ]; then
  "$FV_PYTHON" - <<'PYEOF'
import json
from pathlib import Path
corpus = Path("/mnt/nvme/scratch/sparsefp4_native/t_corpus")
entries = json.loads((corpus / "videos2caption.json").read_text())
n = 8
for s in range(n):
    part = entries[s::n]
    (corpus / f"videos2caption.shard{s}.json").write_text(json.dumps(part, indent=1))
    (corpus / f"merge.shard{s}.txt").write_text(
        f"{corpus / 'videos'},{corpus / f'videos2caption.shard{s}.json'}\n")
    print(f"shard {s}: {len(part)} clips")
PYEOF
  for S in 0 1 2 3 4 5 6 7; do
    ( CUDA_VISIBLE_DEVICES=$S torchrun --nproc_per_node=1 --master_port $((29670 + S)) \
        fastvideo/pipelines/preprocess/v1_preprocess.py \
        --model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
        --data_merge_path "$CORPUS/merge.shard$S.txt" \
        --preprocess_video_batch_size 1 \
        --seed 42 --max_height 480 --max_width 832 --num_frames 77 \
        --dataloader_num_workers 0 \
        --output_dir "$PROC/shard$S/" \
        --train_fps 16 --samples_per_file 8 --flush_frequency 8 \
        --video_length_tolerance_range 5 --preprocess_task "t2v" \
        > "$LOGDIR/preprocess.shard$S.log" 2>&1
      echo "preprocess shard $S rc=$?" ) &
  done
  wait
  mkdir -p "$PROC/combined_parquet_dataset"
  for S in 0 1 2 3 4 5 6 7; do
    SRC="$PROC/shard$S/combined_parquet_dataset/worker_0"
    if [ -d "$SRC" ]; then
      for F in "$SRC"/*.parquet; do
        # hard links (same fs): os.walk does not follow dir symlinks
        ln -f "$F" "$PROC/combined_parquet_dataset/shard${S}_$(basename "$F")"
      done
    fi
  done
  echo "PREPROCESS_DONE"
fi
[ -d "$PROC/combined_parquet_dataset" ] || { echo "T_MATRIX_PREPROCESS_FAILED"; exit 1; }

train_arm() {  # $1 arm, $2 gpus (csv), $3 pipeline, $4 lr, $5 port, extra env via caller
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
    --tracker_project_name "dqvsa_$ARM" \
    --output_dir "checkpoints/dqvsa_$ARM" \
    --max_train_steps 500 \
    --train_batch_size 1 --train_sp_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_latent_t 20 --num_height 480 --num_width 832 --num_frames 77 \
    --enable_gradient_checkpointing_type "full" \
    --learning_rate "$LR" \
    --mixed_precision "bf16" \
    --weight_only_checkpointing_steps 50 \
    --training_state_checkpointing_steps 500 \
    --weight_decay 0.01 --max_grad_norm 1.0 \
    --inference_mode False \
    --checkpoints_total_limit 15 \
    --training_cfg_rate 0.1 \
    --dit_precision "fp32" \
    --ema_start_step 0 --flow_shift 1 --seed 1000 \
    --VSA_decay_rate 0.9 --VSA_decay_interval_steps 1 --VSA_sparsity 0.9 \
    > "$LOGDIR/$ARM.log" 2>&1
  echo "${ARM}_TRAIN_RC=$?"
}

export FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_QAT_VSA256_ATTN
export FASTVIDEO_FA4_BWD_FALLBACK=1

( train_arm T1 0,1 wan_training_pipeline.py          1e-6 29661 ) &
( FASTVIDEO_DQVSA_NAIVE_BWD=1 FASTVIDEO_DQVSA_GRAD_CHECK=1 \
  train_arm T2 2,3 wan_dqvsa_distillation_pipeline.py 1e-5 29662 ) &
( FASTVIDEO_DQVSA_GRAD_CHECK=1 \
  train_arm T3 4,5 wan_dqvsa_distillation_pipeline.py 1e-5 29663 ) &
wait
echo "T_MATRIX_TRAIN_DONE"
