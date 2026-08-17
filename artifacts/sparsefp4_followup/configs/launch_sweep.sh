#!/usr/bin/env bash
# Shard an F1 or F2 sweep over an explicit GPU list, one (seed, prompt) job per GPU
# slot, waves as needed.
#
# Usage:
#   [SEEDS="1234 2026"] launch_sweep.sh <f1|f2> <run-id> "<gpu list>" <n-prompts> [extra runner args...]
# Example:
#   SEEDS="2026 3407" launch_sweep.sh f1 20260816-x-f3a "0 1" 10 --sparsities 0.90
set -euo pipefail

PHASE="${1:?phase (f1|f2) required}"
RUN_ID="${2:?run id required}"
GPU_LIST="${3:?gpu list required, e.g. \"0 1 2 3\"}"
N_PROMPTS="${4:?number of prompts required}"
shift 4
SEEDS="${SEEDS:-1234}"

cd "$(dirname "$0")/../../.."
# shellcheck disable=SC1091
source artifacts/sparsefp4_followup/configs/env.sh

case "$PHASE" in
    f1) RUNNER=artifacts/sparsefp4_followup/configs/f1_run.py ;;
    f2) RUNNER=artifacts/sparsefp4_followup/configs/f2_run.py ;;
    *) echo "unknown phase $PHASE" >&2; exit 2 ;;
esac

LOG_DIR="artifacts/sparsefp4_followup/logs/${RUN_ID}"
mkdir -p "$LOG_DIR"
read -r -a GPUS <<< "$GPU_LIST"
N_GPUS="${#GPUS[@]}"
LAYERS="$(seq -s' ' 0 29)"

echo "phase=$PHASE run_id=$RUN_ID gpus=(${GPUS[*]}) prompts=$N_PROMPTS seeds=($SEEDS) logs=$LOG_DIR"
declare -A PIDS=()
JOB=0

for SEED in $SEEDS; do
    for PROMPT in $(seq 0 $((N_PROMPTS - 1))); do
        SLOT=$((JOB % N_GPUS))
        GPU="${GPUS[$SLOT]}"
        JOB=$((JOB + 1))
        if [ -n "${PIDS[$SLOT]:-}" ]; then
            wait "${PIDS[$SLOT]}" || echo "  WARNING: earlier job on slot $SLOT (gpu $GPU) failed"
        fi
        TAG="p$(printf '%02d' $((PROMPT + 1)))_s${SEED}"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="$GPU" "$FV_PYTHON" "$RUNNER" \
            --run-id "$RUN_ID" \
            --prompt-index "$PROMPT" \
            --seed "$SEED" \
            --steps 50 \
            --timesteps 0 1 10 25 40 48 \
            --layers $LAYERS \
            --cfg-branches positive negative \
            "$@" > "$LOG_DIR/$TAG.log" 2>&1 &
        PIDS[$SLOT]=$!
        echo "  seed $SEED prompt $PROMPT -> gpu $GPU (pid ${PIDS[$SLOT]})"
        sleep 2
    done
done

FAILED=0
for SLOT in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$SLOT]}"; then
        echo "slot $SLOT FAILED"
        FAILED=1
    fi
done

echo "phase=$PHASE run_id=$RUN_ID complete (failed=$FAILED)"
ls -la "$FV_RAW_ROOT/$RUN_ID"/*.jsonl 2>/dev/null | awk '{printf "%9.1f MB  %s\n", $5/1048576, $9}'
exit "$FAILED"
