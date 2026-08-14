#!/usr/bin/env bash
# Phase 5 main sweep: 10 prompts x 6 arms at one matched sparsity, sharded across
# 8 B200s, one process per generation at sp_size=1. Each generation gets its own
# process because the arm is fixed at model-load time, which also yields a clean
# per-arm peak-memory number.
#
# Videos and float frames go to /mnt/scratch (the root volume has <5 GiB free);
# only contact sheets, metrics and a few sample videos are copied into
# artifacts/sparsefp4/ later.
#
#   bash artifacts/sparsefp4/configs/phase5_launch.sh <run_id> [sparsity] [seed]
set -euo pipefail

RUN_ID="${1:?usage: phase5_launch.sh <run_id> [sparsity] [seed]}"
SPARSITY="${2:-0.90}"
SEED="${3:-1234}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source artifacts/sparsefp4/configs/env.sh

RAW_ROOT="${RAW_ROOT:-/mnt/scratch/sparsefp4}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/scratch/sparsefp4-videos}"
LOG_DIR="artifacts/sparsefp4/logs/${RUN_ID}"
STEPS="${STEPS:-50}"
STAGE="${STAGE:-5-main}"
PROMPTS="${PROMPTS:-0 1 2 3 4 5 6 7 8 9}"
ARMS="${ARMS:-DENSE-BF16 DENSE-FP4 SPARSE-BF16 SPARSE-FP4-NAIVE SPARSE-FP4-ROUTE8 SPARSE-FP4-ROUTE16}"
NGPU="${NGPU:-8}"

mkdir -p "$LOG_DIR" "$RAW_ROOT/$RUN_ID" "$VIDEO_ROOT/$RUN_ID"

# Build the full job list, then deal it round-robin to GPUs. Arms are the inner
# loop so every GPU sees a mix of cheap dense and expensive sparse work and the
# shards finish at roughly the same time.
JOBS=()
for idx in $PROMPTS; do
  for arm in $ARMS; do
    JOBS+=("$idx:$arm:0")
  done
done

# Perturbation calibration ladder (SPARSE-BF16-EPS). Answers the question the
# free-running pixel differences cannot: how much final-video difference does a
# *known* per-call attention perturbation produce? If the curve saturates well
# below the routing-precision perturbation magnitude, the pixel metrics are
# measuring trajectory divergence rather than attention error, and must be
# reported as saturated. Run on a prompt subset -- the shape of the curve is the
# result, not its per-prompt spread.
EPS_LADDER="${EPS_LADDER:-1e-5 1e-4 1e-3 1e-2 1e-1}"
EPS_PROMPTS="${EPS_PROMPTS:-0 4}"
if [[ "${WITH_EPS_LADDER:-1}" == "1" ]]; then
  for idx in $EPS_PROMPTS; do
    for eps in $EPS_LADDER; do
      JOBS+=("$idx:SPARSE-BF16-EPS:$eps")
    done
  done
fi

echo "phase5: run_id=$RUN_ID jobs=${#JOBS[@]} sparsity=$SPARSITY seed=$SEED steps=$STEPS gpus=$NGPU"

for gpu in $(seq 0 $((NGPU - 1))); do
  (
    for job_index in $(seq "$gpu" "$NGPU" $((${#JOBS[@]} - 1))); do
      job="${JOBS[$job_index]}"
      IFS=':' read -r idx arm eps <<< "$job"
      pid_tag="p$(printf '%02d' $((idx + 1)))"
      log_tag="${pid_tag}_${arm}_s${SEED}"
      [[ "$eps" != "0" ]] && log_tag="${log_tag}_eps${eps}"
      CUDA_VISIBLE_DEVICES="$gpu" "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_run.py \
        --run-id "$RUN_ID" --prompt-index "$idx" --arm "$arm" \
        --sparsity "$SPARSITY" --seed "$SEED" --steps "$STEPS" --stage "$STAGE" \
        --perturb-rel-l2 "$eps" \
        --raw-root "$RAW_ROOT" --video-root "$VIDEO_ROOT" \
        > "$LOG_DIR/gpu${gpu}_${log_tag}.log" 2>&1 \
        || echo "FAILED gpu=$gpu prompt=$idx arm=$arm eps=$eps seed=$SEED" >> "$LOG_DIR/FAILURES.txt"
    done
    echo "gpu $gpu done"
  ) &
done

wait
echo "PHASE5_LAUNCH_COMPLETE $RUN_ID"
