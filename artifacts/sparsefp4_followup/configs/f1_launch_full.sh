#!/usr/bin/env bash
# Shard an F1 run across the 8 B200s: one prompt per GPU, two waves for 10 prompts.
#
# Usage: f1_launch_full.sh <run-id> [extra args passed to f1_run.py]
set -euo pipefail

RUN_ID="${1:?run id required}"
shift || true

cd "$(dirname "$0")/../../.."
# shellcheck disable=SC1091
source artifacts/sparsefp4_followup/configs/env.sh

LOG_DIR="artifacts/sparsefp4_followup/logs/${RUN_ID}"
mkdir -p "$LOG_DIR"
N_GPUS="$(nvidia-smi --list-gpus | wc -l)"
LAYERS="$(seq -s' ' 0 29)"

echo "run_id=$RUN_ID gpus=$N_GPUS logs=$LOG_DIR"

for PROMPT in $(seq 0 9); do
    GPU=$((PROMPT % N_GPUS))
    if [ "$PROMPT" -ge "$N_GPUS" ]; then
        # Second wave: wait for the GPU's first-wave job before reusing it.
        wait "${PIDS[$GPU]}"
    fi
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$GPU" "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f1_run.py \
        --run-id "$RUN_ID" \
        --prompt-index "$PROMPT" \
        --steps 50 \
        --timesteps 0 1 10 25 40 48 \
        --layers $LAYERS \
        --cfg-branches positive negative \
        --sparsities 0.80 0.90 0.95 \
        --stage F1-full \
        "$@" > "$LOG_DIR/p$(printf '%02d' $((PROMPT + 1))).log" 2>&1 &
    PIDS[$GPU]=$!
    echo "  launched prompt $PROMPT on GPU $GPU (pid ${PIDS[$GPU]})"
    sleep 3
done

FAILED=0
for GPU in $(seq 0 $((N_GPUS - 1))); do
    if [ -n "${PIDS[$GPU]:-}" ]; then
        if ! wait "${PIDS[$GPU]}"; then
            echo "GPU $GPU job FAILED"
            FAILED=1
        fi
    fi
done

echo "all jobs done (failed=$FAILED)"
ls -la "$FV_RAW_ROOT/$RUN_ID"/*.jsonl | awk '{print $5, $9}'
exit "$FAILED"
