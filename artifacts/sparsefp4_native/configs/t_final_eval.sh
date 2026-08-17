#!/usr/bin/env bash
# Paper-scale (326-prompt) paired evaluation of the two best DQ-VSA gates:
# T3-c250 and T3-c500 served through the NATIVE P4 path, scored with the
# same dimension-routed VBench + paired protocol as the V2 arms, then
# Holm-corrected paired bootstrap vs P4G (teacher) and vs untrained P4.
set -u
REPO=/home/ec2-user/FastVideo
cd "$REPO"
source artifacts/sparsefp4_followup/configs/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR="$REPO/artifacts/sparsefp4_native/logs/t_final"
mkdir -p "$LOGDIR"
SERVE=/mnt/nvme/scratch/sparsefp4_native/t_serve
VIDEOS=/mnt/nvme/scratch/sparsefp4_native/paper_videos/paper-s090

# ---- generation: 2 arms x 8 shards, 4 GPUs per arm ----
gen_arm() {  # $1 label, $2 serve-dir, $3 gpu-base (0 or 4)
  local LABEL=$1 DIR=$2 BASE=$3
  for K in 0 1 2 3; do
    ( for HALF in 0 1; do
        S=$((K + 4 * HALF))
        CUDA_VISIBLE_DEVICES=$((BASE + K)) "$FV_PYTHON" \
          "$REPO/artifacts/sparsefp4_native/configs/paper_run.py" \
          --arm P4 --arm-label "$LABEL" --model-path "$DIR" \
          --shard $S --num-shards 8 --run-id paper-s090 \
          > "$LOGDIR/gen_${LABEL}_s$S.log" 2>&1
        echo "[$LABEL shard $S] rc=$?"
      done ) &
  done
}
gen_arm T3c250 "$SERVE/T3-c250" 0
gen_arm T3c500 "$SERVE/T3-c500" 4
wait
echo "T_FINAL_GEN_DONE"

# ---- scoring: vbench dims sharded over GPUs + paired metrics ----
export PAPER_SCORE_ARMS="T3c250,T3c500"
for S in 0 1 2 3; do
  ( CUDA_VISIBLE_DEVICES=$S "$FV_PYTHON" \
      "$REPO/artifacts/sparsefp4_native/configs/paper_score.py" \
      --videos "$VIDEOS" --what vbench --shard $S --num-shards 4 \
      --out-tag t_final_vbench \
      > "$LOGDIR/score_vb$S.log" 2>&1
    echo "vb shard $S rc=$?" ) &
done
for S in 0 1 2 3; do
  ( CUDA_VISIBLE_DEVICES=$((4 + S)) PAPER_SCORE_ARMS="P0,T3c250,T3c500" "$FV_PYTHON" \
      "$REPO/artifacts/sparsefp4_native/configs/paper_score.py" \
      --videos "$VIDEOS" --what paired --shard $S --num-shards 4 \
      --out-tag t_final_paired \
      > "$LOGDIR/score_px$S.log" 2>&1
    echo "px shard $S rc=$?" ) &
done
wait
echo "T_FINAL_SCORE_DONE"
