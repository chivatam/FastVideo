#!/usr/bin/env bash
# P4/P4G quality (10 prompts each) + perf reps.
set -u
REPO=/home/ec2-user/FastVideo
source "$REPO/artifacts/sparsefp4_followup/configs/env.sh"
LOGDIR="$REPO/artifacts/sparsefp4_native/logs/p4_runs"
mkdir -p "$LOGDIR"
RUN=pq-s090   # same run-id so videos land beside the other arms

quality_queue() {  # $1 gpu, $2 arm, prompts...
  local GPU=$1 ARM=$2; shift 2
  for P in "$@"; do
    CUDA_VISIBLE_DEVICES=$GPU "$FV_PYTHON" \
      "$REPO/artifacts/sparsefp4_native/configs/p_run.py" \
      --run-id $RUN --arm "$ARM" --prompt-index "$P" \
      > "$LOGDIR/p$(printf %02d "$P")_${ARM}.log" 2>&1
    echo "[gpu$GPU] $ARM p$P rc=$?"
  done
}

# perf reps on dedicated GPUs
( CUDA_VISIBLE_DEVICES=1 "$FV_PYTHON" "$REPO/artifacts/sparsefp4_native/configs/p_run.py" \
    --run-id perf-s090 --arm P4 --prompt-index 0 --perf-reps 5 \
    > "$LOGDIR/perf_P4.log" 2>&1; echo "perf P4 rc=$?" ) &
( CUDA_VISIBLE_DEVICES=3 "$FV_PYTHON" "$REPO/artifacts/sparsefp4_native/configs/p_run.py" \
    --run-id perf-s090 --arm P4G --prompt-index 0 --perf-reps 5 \
    > "$LOGDIR/perf_P4G.log" 2>&1; echo "perf P4G rc=$?" ) &

# quality: 20 jobs over 4 GPUs
( quality_queue 4 P4 0 1 2 3 4 ) &
( quality_queue 5 P4 5 6 7 8 9 ) &
( quality_queue 6 P4G 0 1 2 3 4 ) &
( quality_queue 7 P4G 5 6 7 8 9 ) &
wait
echo "P4_ALL_DONE"
