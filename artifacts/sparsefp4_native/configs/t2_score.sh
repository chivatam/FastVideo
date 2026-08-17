#!/usr/bin/env bash
# Score the T2-c250 paper-scale generations (vbench dims + paired vs P0).
set -u
REPO=/home/ec2-user/FastVideo
cd "$REPO"
source artifacts/sparsefp4_followup/configs/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR="$REPO/artifacts/sparsefp4_native/logs/t_final"
VIDEOS=/mnt/nvme/scratch/sparsefp4_native/paper_videos/paper-s090
mkdir -p "$LOGDIR"

for S in 0 1 2 3; do
  ( CUDA_VISIBLE_DEVICES=$S PAPER_SCORE_ARMS="T2c250" "$FV_PYTHON" \
      "$REPO/artifacts/sparsefp4_native/configs/paper_score.py" \
      --videos "$VIDEOS" --what vbench --shard $S --num-shards 4 \
      --out-tag t2_final_vbench \
      > "$LOGDIR/t2_score_vb$S.log" 2>&1
    echo "t2 vb shard $S rc=$?" ) &
done
for S in 0 1 2 3; do
  ( CUDA_VISIBLE_DEVICES=$((4 + S)) PAPER_SCORE_ARMS="P0,T2c250" "$FV_PYTHON" \
      "$REPO/artifacts/sparsefp4_native/configs/paper_score.py" \
      --videos "$VIDEOS" --what paired --shard $S --num-shards 4 \
      --out-tag t2_final_paired \
      > "$LOGDIR/t2_score_px$S.log" 2>&1
    echo "t2 px shard $S rc=$?" ) &
done
wait
echo "T2_SCORE_DONE"
