#!/usr/bin/env bash
# T-matrix gate evaluation: serve every (T-arm, gate) checkpoint through the
# NATIVE P4 path and generate the 10 dev prompts, then score.
set -u
REPO=/home/ec2-user/FastVideo
cd "$REPO"
source artifacts/sparsefp4_followup/configs/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR="$REPO/artifacts/sparsefp4_native/logs/t_gates"
mkdir -p "$LOGDIR"

HF_SNAP=$(ls -d "$FV_SCRATCH"/hf-cache/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/*/ | head -1)
SERVE_ROOT=/mnt/nvme/scratch/sparsefp4_native/t_serve

serve_dir() {  # $1 arm, $2 gate
  local D="$SERVE_ROOT/$1-c$2"
  mkdir -p "$D"
  for c in model_index.json scheduler text_encoder tokenizer vae; do
    ln -sfn "$HF_SNAP/$c" "$D/$c"
  done
  ln -sfn "$REPO/checkpoints/dqvsa_$1/checkpoint-$2/transformer" "$D/transformer"
  echo "$D"
}

# job list: (arm, gate, prompt) round-robined over 8 GPUs
JOBS=()
for ARM in T1 T2 T3; do
  for GATE in 100 250 500; do
    [ -d "$REPO/checkpoints/dqvsa_$ARM/checkpoint-$GATE/transformer" ] || {
      echo "MISSING checkpoint dqvsa_$ARM/checkpoint-$GATE"; continue; }
    serve_dir "$ARM" "$GATE" > /dev/null
    for P in 0 1 2 3 4 5 6 7 8 9; do JOBS+=("$ARM $GATE $P"); done
  done
done
echo "gate-eval jobs: ${#JOBS[@]}"

run_queue() {  # $1 gpu; consumes every 8th job starting at $1
  local GPU=$1
  local i
  for ((i=GPU; i<${#JOBS[@]}; i+=8)); do
    set -- ${JOBS[$i]}
    local ARM=$1 GATE=$2 P=$3
    local TAGDIR="tgate-$ARM-c$GATE"
    CUDA_VISIBLE_DEVICES=$GPU "$FV_PYTHON" \
      "$REPO/artifacts/sparsefp4_native/configs/p_run.py" \
      --run-id "$TAGDIR" --arm P4 --prompt-index "$P" \
      --model-path "$SERVE_ROOT/$ARM-c$GATE" \
      > "$LOGDIR/${TAGDIR}_p$(printf %02d "$P").log" 2>&1
    echo "[gpu$GPU] $TAGDIR p$P rc=$?"
  done
}

for GPU in 0 1 2 3 4 5 6 7; do ( run_queue $GPU ) & done
wait
echo "T_GATE_GEN_DONE"

# ---- scoring: VBench dims sharded over 7 GPUs, pixel metrics on GPU 7 ----
for S in 0 1 2 3 4 5 6; do
  ( CUDA_VISIBLE_DEVICES=$S "$FV_PYTHON" \
      "$REPO/artifacts/sparsefp4_native/configs/t_gate_score.py" \
      --shard $S --num-shards 7 > "$LOGDIR/score_vb$S.log" 2>&1
    echo "score shard $S rc=$?" ) &
done
( CUDA_VISIBLE_DEVICES=7 "$FV_PYTHON" \
    "$REPO/artifacts/sparsefp4_native/configs/t_gate_score.py" --pixel \
    > "$LOGDIR/score_pixel.log" 2>&1
  echo "score pixel rc=$?" ) &
wait
"$FV_PYTHON" "$REPO/artifacts/sparsefp4_native/configs/t_gate_score.py" --aggregate
echo "T_GATE_EVAL_DONE"
