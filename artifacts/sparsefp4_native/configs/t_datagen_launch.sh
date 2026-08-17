#!/usr/bin/env bash
# T-matrix corpus generation: 1290 clips over 8 GPUs, then finalize.
set -u
REPO=/home/ec2-user/FastVideo
cd "$REPO"
source artifacts/sparsefp4_followup/configs/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR="$REPO/artifacts/sparsefp4_native/logs/t_datagen"
mkdir -p "$LOGDIR"

for GPU in 0 1 2 3 4 5 6 7; do
  ( CUDA_VISIBLE_DEVICES=$GPU "$FV_PYTHON" \
      "$REPO/artifacts/sparsefp4_native/configs/t_datagen.py" \
      --shard $GPU --num-shards 8 \
      > "$LOGDIR/shard$GPU.log" 2>&1
    echo "shard $GPU rc=$?" ) &
done
wait
"$FV_PYTHON" "$REPO/artifacts/sparsefp4_native/configs/t_datagen.py" --finalize
echo "DATAGEN_ALL_DONE"
