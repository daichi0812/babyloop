"""本番ランの max_steps / warmup を「語数目標 × 実トークン比」から算出する。

max_steps は cosine 減衰のホライズンなので、桁を固定せず **毎回** ``max_words_seen``
と前処理メタの実比 ``total_tokens / total_words`` から出し直す。

使い方（前処理後に）:
    uv run python scripts/plan_run.py --max-words-seen 1000000000              # 本番1B
    uv run python scripts/plan_run.py --max-words-seen 100000000               # lr probe(1ep)
    uv run python scripts/plan_run.py --max-words-seen 1000000000 --grad-accum-steps 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/text/tokenized/meta.json",
                    help="前処理メタ(total_words/total_tokens)")
    ap.add_argument("--max-words-seen", type=int, required=True,
                    help="目標の累積入力語数（露出。本番上限=1_000_000_000）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--warmup-frac", type=float, default=0.015, help="warmup = max_steps の割合(1〜2%)")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    total_words = int(meta["total_words"])
    total_tokens = int(meta["total_tokens"])
    ratio = total_tokens / total_words  # tokens per word（実測）

    eff_batch_tokens = args.batch_size * args.grad_accum_steps * args.max_seq_len
    tokens_target = args.max_words_seen * ratio
    max_steps = round(tokens_target / eff_batch_tokens)
    warmup_steps = max(1, round(max_steps * args.warmup_frac))

    print(f"meta: total_words={total_words:,} total_tokens={total_tokens:,}")
    print(f"tokens/word (実比)        = {ratio:.4f}")
    print(f"effective batch (tokens)  = {args.batch_size}×{args.grad_accum_steps}×{args.max_seq_len} = {eff_batch_tokens:,}")
    print(f"max_words_seen 目標        = {args.max_words_seen:,}  (= tokens {round(tokens_target):,})")
    print(f"=> max_steps              = {max_steps:,}")
    print(f"=> warmup_steps ({args.warmup_frac:.1%}) = {warmup_steps:,}")
    print()
    print("train.py への override:")
    print(
        f"  train.grad_accum_steps={args.grad_accum_steps} "
        f"train.max_words_seen={args.max_words_seen} "
        f"train.max_steps={max_steps} train.warmup_steps={warmup_steps}"
    )


if __name__ == "__main__":
    main()
