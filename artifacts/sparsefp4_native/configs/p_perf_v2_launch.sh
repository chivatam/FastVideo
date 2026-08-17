#!/usr/bin/env bash
# V2 canonical performance runs: all arms x both resolutions under ONE
# allocator configuration (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True).
# Replaces the mixed pre/post-allocator-fix numbers in c8_performance.md.
set -u
REPO=/home/ec2-user/FastVideo
source "$REPO/artifacts/sparsefp4_followup/configs/env.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR="$REPO/artifacts/sparsefp4_native/logs/perf_v2"
mkdir -p "$LOGDIR"

run_arm() {  # $1 gpu, $2 run-id, $3 arm, $4 height, $5 width
  local GPU=$1 RUN=$2 ARM=$3 H=$4 W=$5
  CUDA_VISIBLE_DEVICES=$GPU "$FV_PYTHON" \
    "$REPO/artifacts/sparsefp4_native/configs/p_run.py" \
    --run-id "$RUN" --arm "$ARM" --prompt-index 0 --perf-reps 3 \
    --height "$H" --width "$W" \
    > "$LOGDIR/${RUN}_${ARM}.log" 2>&1
  echo "[gpu$GPU] $RUN $ARM rc=$?"
}

# 720p arms, one GPU each
( run_arm 0 perf720-v2 P0  720 1280 ) &
( run_arm 1 perf720-v2 P1  720 1280 ) &
( run_arm 2 perf720-v2 P2  720 1280 ) &
( run_arm 3 perf720-v2 P4G 720 1280 ) &
( run_arm 4 perf720-v2 P4  720 1280 ) &
# 480p arms (default resolution), queued on remaining GPUs
( run_arm 5 perf480-v2 P0 0 0; run_arm 5 perf480-v2 P1 0 0 ) &
( run_arm 6 perf480-v2 P2 0 0; run_arm 6 perf480-v2 P4G 0 0 ) &
( run_arm 7 perf480-v2 P4 0 0 ) &
wait
echo "PERF_V2_ALL_DONE"
