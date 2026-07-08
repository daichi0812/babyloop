#!/bin/bash
# per-task 3seed化: zero-shot 6タスク × 13 model-seed (GPU1)
export PATH="$HOME/.local/bin:$PATH"
PROJ=~/dev-hub/research/ml/babylm/babyloop
cd $PROJ/third_party/evaluation-pipeline/strict
export CUDA_VISIBLE_DEVICES=1
MODELS="babyloop-std-text-s43 babyloop-std-text-s44 \
babyloop-loop-text-s43 babyloop-loop-text-s44 \
babyloop-std-mm-s43 babyloop-std-mm-s44 \
babyloop-loop-mm-s42 babyloop-loop-mm-s44 \
babyloop-std-mm-id-s42 babyloop-std-mm-id-s43 babyloop-std-mm-id-s44 \
babyloop-loop-mm-prefix-s42 babyloop-loop-mm-prefix-s43"
for M in $MODELS; do
  echo "===== [$(date -u +%H:%M)] START $M ====="
  uv run --project $PROJ --group eval bash scripts/eval_zero_shot.sh eval_models/$M causal
  echo "===== [$(date -u +%H:%M)] DONE $M ====="
done
echo "ALL DONE $(date -u)"
