#!/usr/bin/env bash
# Phase 2 main sweep: 10 prompts sharded across 8 B200s, one process per GPU at
# sp_size=1. Each process writes its own JSONL shard and its own log, so a
# dropped session loses nothing; poll the logs rather than blocking.
#
#   bash artifacts/sparsefp4/configs/phase2_launch.sh <run_id>
set -euo pipefail

RUN_ID="${1:?usage: phase2_launch.sh <run_id>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source artifacts/sparsefp4/configs/env.sh

RAW_ROOT="${RAW_ROOT:-/mnt/scratch/sparsefp4}"
LOG_DIR="artifacts/sparsefp4/logs/${RUN_ID}"
mkdir -p "$LOG_DIR" "$RAW_ROOT/$RUN_ID"

STEPS="${STEPS:-50}"
SPARSITIES="${SPARSITIES:-0.80 0.90 0.95}"
LAYERS="${LAYERS:-0 1 2 5 6 8 10 11 13 16 20 23 24 25 27 28 29}"
TIMESTEPS="${TIMESTEPS:-0 1 10 25 40}"
MECH_LAYERS="${MECH_LAYERS:-0 5 13 24 28 29}"
MECH_TIMESTEPS="${MECH_TIMESTEPS:-0 25}"
MECH_QBLOCKS="${MECH_QBLOCKS:-12}"

# GPU -> prompt index list. Eight GPUs, ten prompts: the first two GPUs take a
# second prompt sequentially.
declare -a ASSIGN=( "0 8" "1 9" "2" "3" "4" "5" "6" "7" )

for gpu in "${!ASSIGN[@]}"; do
  prompts="${ASSIGN[$gpu]}"
  (
    for idx in $prompts; do
      CUDA_VISIBLE_DEVICES="$gpu" "$FV_PYTHON" artifacts/sparsefp4/configs/phase2_run.py \
        --run-id "$RUN_ID" --prompt-index "$idx" \
        --sparsities $SPARSITIES --layers $LAYERS --timesteps $TIMESTEPS \
        --mechanism-layers $MECH_LAYERS --mechanism-timesteps $MECH_TIMESTEPS \
        --mechanism-sparsities $SPARSITIES --mechanism-query-blocks "$MECH_QBLOCKS" \
        --score-dtype float64 --steps "$STEPS" --stage 2-main \
        --raw-root "$RAW_ROOT" \
        >> "$LOG_DIR/gpu${gpu}_p$(printf '%02d' $((idx + 1))).log" 2>&1
    done
  ) &
  echo "launched gpu=$gpu prompts=[$prompts]"
done

wait
echo "PHASE2_LAUNCH_COMPLETE $RUN_ID"
