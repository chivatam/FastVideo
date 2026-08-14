#!/usr/bin/env bash
# Phase 2B geometry control: re-run the decisive subset at the geometries the
# deployed sparse backend actually uses. Two new geometry arms, 10 prompts each,
# sharded across 8 B200s at sp_size=1. Each process writes its own JSONL shard
# and its own log, so a dropped session loses nothing; poll the logs rather than
# blocking.
#
#   bash artifacts/sparsefp4/configs/phase2b_launch.sh <run_id> <geometry>
#
# geometry is one of 64x64-raster | 64x64-cube | 128x64-raster. The 128x64-raster
# arm already exists as run 20260814-025500-8208536-p2-main and is not re-run.
#
# Arms: A (dense BF16 reference), C (BF16 mask), C_null (bf16-vs-bf16 null
# control, kept in every run), D (NVFP4 mask), C_rand (equal-magnitude random
# contrast). B/B_sim/D8/E/F8/F16 are omitted: Phase 2 already established the
# quantization and H3 arms at 128x64, and this control is about the mask, not the
# compute precision. No latency claim is made from any of it.
set -euo pipefail

RUN_ID="${1:?usage: phase2b_launch.sh <run_id> <geometry>}"
GEOMETRY="${2:?usage: phase2b_launch.sh <run_id> <geometry>}"
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
ARMS="${ARMS:-A C C_null D C_rand}"
# Emitted once per measured cell, at all three geometries, so the
# Phase-1-vs-Phase-2 tie-count discrepancy can be settled from one run's records.
TIE_GEOMETRIES="${TIE_GEOMETRIES:-128x64-raster 64x64-raster 64x64-cube}"

# GPU -> prompt index list. Eight GPUs, ten prompts: the first two GPUs take a
# second prompt sequentially.
declare -a ASSIGN=( "0 8" "1 9" "2" "3" "4" "5" "6" "7" )

for gpu in "${!ASSIGN[@]}"; do
  prompts="${ASSIGN[$gpu]}"
  (
    for idx in $prompts; do
      CUDA_VISIBLE_DEVICES="$gpu" "$FV_PYTHON" artifacts/sparsefp4/configs/phase2_run.py \
        --run-id "$RUN_ID" --prompt-index "$idx" \
        --geometry "$GEOMETRY" --arms $ARMS \
        --tie-diagnostic-geometries $TIE_GEOMETRIES --no-activation-stats \
        --sparsities $SPARSITIES --layers $LAYERS --timesteps $TIMESTEPS \
        --mechanism-layers $MECH_LAYERS --mechanism-timesteps $MECH_TIMESTEPS \
        --mechanism-sparsities $SPARSITIES --mechanism-query-blocks "$MECH_QBLOCKS" \
        --score-dtype float64 --steps "$STEPS" --stage "2b-${GEOMETRY}" \
        --raw-root "$RAW_ROOT" \
        >> "$LOG_DIR/gpu${gpu}_p$(printf '%02d' $((idx + 1))).log" 2>&1
    done
  ) &
  echo "launched gpu=$gpu prompts=[$prompts] geometry=$GEOMETRY"
done

wait
echo "PHASE2B_LAUNCH_COMPLETE $RUN_ID $GEOMETRY"
