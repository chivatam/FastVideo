#!/usr/bin/env bash
# Stage 2 of SparseFP4 Phase 1: 10 prompts x 1 seed x full sparsity sweep,
# sharded one process per GPU at sp_size=1 so head indices stay global.
#
# Each shard is detached with nohup and streams to its own log under
# artifacts/sparsefp4/logs/, so progress survives a dropped session and can be
# polled instead of held open in a foreground call.
#
# Usage:
#   source artifacts/sparsefp4/configs/env.sh
#   bash artifacts/sparsefp4/configs/phase1_stage2_launch.sh <run_id>
set -euo pipefail

RUN_ID="${1:?usage: phase1_stage2_launch.sh <run_id>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DRIVER="$REPO_ROOT/artifacts/sparsefp4/configs/phase1_probe_run.py"
LOG_DIR="$REPO_ROOT/artifacts/sparsefp4/logs"
RAW_ROOT="${SPARSEFP4_RAW_ROOT:-/mnt/scratch/sparsefp4}"

# Volume control, recorded in every shard's probe config so the subsampling rule
# is part of the archived run description (EXPERIMENT_SPEC 11 rule 2): the full
# 50-step x 5-sparsity x 10-prompt enumeration projects ~8 GB of JSONL, past the
# spec's 5 GiB retained-raw cap, so timesteps are decimated on a deterministic
# lattice while every layer and head stays enumerated.
MEASURE_TIMESTEP_STRIDE="${MEASURE_TIMESTEP_STRIDE:-5}"
NULL_LAYER_STRIDE="${NULL_LAYER_STRIDE:-5}"
NULL_TIMESTEP_STRIDE="${NULL_TIMESTEP_STRIDE:-10}"
SPARSITIES="${SPARSITIES:-0.50 0.70 0.80 0.90 0.95}"
STEPS="${STEPS:-50}"

# GPU -> prompt indices (0-based). 10 prompts over 8 GPUs: GPU 0 and 1 each run a
# second prompt sequentially once their first finishes. Never two processes on one
# GPU at the same time.
GPU_PROMPTS=(
  "0 8"
  "1 9"
  "2"
  "3"
  "4"
  "5"
  "6"
  "7"
)

mkdir -p "$LOG_DIR"
echo "run_id=$RUN_ID raw_root=$RAW_ROOT measure_timestep_stride=$MEASURE_TIMESTEP_STRIDE"

for gpu in "${!GPU_PROMPTS[@]}"; do
  prompts="${GPU_PROMPTS[$gpu]}"
  log="$LOG_DIR/stage2_${RUN_ID}_gpu${gpu}.log"
  # One detached bash per GPU walks its prompt list sequentially.
  nohup bash -c "
    for prompt_index in $prompts; do
      echo \"=== SHARD gpu=$gpu prompt_index=\$prompt_index start \$(date -u +%FT%TZ) ===\"
      CUDA_VISIBLE_DEVICES=$gpu '$FV_PYTHON' '$DRIVER' \
        --run-id '$RUN_ID' \
        --prompt-index \"\$prompt_index\" \
        --sparsities $SPARSITIES \
        --routing-precisions bf16 fp8_e4m3 nvfp4 nvfp4_sim \
        --steps $STEPS \
        --stage 2 \
        --measure-timestep-stride $MEASURE_TIMESTEP_STRIDE \
        --null-control-layer-stride $NULL_LAYER_STRIDE \
        --null-control-timestep-stride $NULL_TIMESTEP_STRIDE \
        --spearman-timestep-stride 10 \
        --raw-root '$RAW_ROOT' || echo \"SHARD FAILED gpu=$gpu prompt_index=\$prompt_index rc=\$?\"
      echo \"=== SHARD gpu=$gpu prompt_index=\$prompt_index done \$(date -u +%FT%TZ) ===\"
    done
    echo 'ALL SHARDS ON THIS GPU DONE'
  " > "$log" 2>&1 &
  echo "launched gpu=$gpu prompts=[$prompts] pid=$! log=$log"
done

wait
echo "all stage 2 shards finished"
