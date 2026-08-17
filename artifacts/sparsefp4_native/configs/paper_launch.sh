#!/usr/bin/env bash
# Paper-scale sweep: 5 arms x 326 prompts, arms run sequentially, each arm
# sharded over all 8 GPUs. ~35 min/arm -> ~3h total.
set -u
REPO=/home/ec2-user/FastVideo
cd "$REPO"
source artifacts/sparsefp4_followup/configs/env.sh
LOGDIR="$REPO/artifacts/sparsefp4_native/logs/paper_runs"
mkdir -p "$LOGDIR"
RUN_ID=${1:-paper-s090}

for ARM in P0 P1 P2 P4G P4; do
  echo "=== arm $ARM start $(date +%H:%M:%S)"
  for GPU in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$GPU "$FV_PYTHON" \
      "$REPO/artifacts/sparsefp4_native/configs/paper_run.py" \
      --arm "$ARM" --shard "$GPU" --num-shards 8 --run-id "$RUN_ID" \
      > "$LOGDIR/${ARM}_shard${GPU}.log" 2>&1 &
  done
  wait
  echo "=== arm $ARM done $(date +%H:%M:%S)"
done
echo "PAPER_SWEEP_DONE"
