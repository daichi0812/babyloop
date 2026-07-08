#!/bin/bash
# featoff BLiMP eval ×3 (evaluate.py, 他セルと同一経路) → zero-shot 4 model-seed / GPU0
export PATH="$HOME/.local/bin:$PATH"
PROJ=~/dev-hub/research/ml/babylm/babyloop
export CUDA_VISIBLE_DEVICES=0
cd $PROJ
for S in 42 43 44; do
  echo "===== [$(date -u +%H:%M)] BLIMP-EVAL std_mm_featoff seed=$S ====="
  uv run python scripts/evaluate.py experiment=std_mm_featoff \
    +checkpoint=outputs/std_mm_featoff/seed_$S/ckpt_final +eval_split=full
done
cd $PROJ/third_party/evaluation-pipeline/strict
for M in babyloop-std-mm-featoff-s42 babyloop-std-mm-featoff-s43 \
         babyloop-std-mm-featoff-s44 babyloop-loop-mm-prefix-s44; do
  echo "===== [$(date -u +%H:%M)] ZERO-SHOT $M ====="
  uv run --project $PROJ --group eval bash scripts/eval_zero_shot.sh eval_models/$M causal
done
echo "ALL DONE $(date -u)"
