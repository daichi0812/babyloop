#!/bin/bash
# features-off 対照 (std_mm_featoff) 3seed 逐次 / GPU1
export PATH="$HOME/.local/bin:$PATH"
PROJ=~/dev-hub/research/ml/babylm/babyloop
cd $PROJ
export CUDA_VISIBLE_DEVICES=1
mkdir -p runs/std_mm_featoff
for S in 42 43 44; do
  echo "===== [$(date -u +%F' '%H:%M)] START std_mm_featoff seed=$S ====="
  uv run python scripts/train.py experiment=std_mm_featoff \
    train.seed=$S train.lr=1.5e-3 train.grad_accum_steps=8 \
    train.max_words_seen=1000000000 train.max_steps=20000 \
    train.save_state_every=200 \
    2>&1 | tee runs/std_mm_featoff/launch_seed_$S.log
  echo "===== [$(date -u +%F' '%H:%M)] DONE std_mm_featoff seed=$S ====="
done
echo "ALL DONE $(date -u)"
