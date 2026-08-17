#!/usr/bin/env bash
# Launch the P-arm quality sweep: 5 arms x 10 prompts, sharded over GPUs.
#   artifacts/sparsefp4_native/configs/p_launch.sh <run-id> "4 5 6 7"
set -u
RUN_ID=${1:?run-id}
GPUS=(${2:?gpu list})
REPO=/home/ec2-user/FastVideo
source "$REPO/artifacts/sparsefp4_followup/configs/env.sh"
LOGDIR="$REPO/artifacts/sparsefp4_native/logs/p_runs_$RUN_ID"
mkdir -p "$LOGDIR"

JOBS=()
for ARM in P0 P1 P2 P2G P3; do
  for P in 0 1 2 3 4 5 6 7 8 9; do
    JOBS+=("$ARM $P")
  done
done

NGPU=${#GPUS[@]}
i=0
for JOB in "${JOBS[@]}"; do
  GPU=${GPUS[$((i % NGPU))]}
  echo "$JOB $GPU"
  i=$((i + 1))
done | awk '{print $1, $2, $3}' > "$LOGDIR/schedule.txt"

# run each GPU's queue sequentially, all GPUs in parallel
for GPU in "${GPUS[@]}"; do
  (
    grep " $GPU\$" "$LOGDIR/schedule.txt" | while read -r ARM P _; do
      TAG="p$(printf %02d "$P")_${ARM}"
      echo "[gpu$GPU] start $TAG $(date +%H:%M:%S)"
      CUDA_VISIBLE_DEVICES=$GPU "$FV_PYTHON" \
        "$REPO/artifacts/sparsefp4_native/configs/p_run.py" \
        --run-id "$RUN_ID" --arm "$ARM" --prompt-index "$P" \
        > "$LOGDIR/${TAG}.log" 2>&1
      rc=$?
      echo "[gpu$GPU] done  $TAG rc=$rc $(date +%H:%M:%S)"
    done
  ) &
done
wait
echo "ALL P-ARM JOBS COMPLETE"
