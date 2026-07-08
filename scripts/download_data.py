"""BabyLM 2026 学習コーパスの取得。

一次ターゲットは BabyLM 2026 Strict（text-only, 100M words）の HF dataset
``BabyLM-community/BabyLM-2026-Strict``。1行=1サンプルで ``data/text/train.txt``
へ書き出す。再現性のため出所・版・取得日を ``data/text/SOURCE.json`` に残す
（docs/reproduce.md 要件）。

使い方:
    uv run python scripts/download_data.py                     # 全量(約543MB)
    uv run python scripts/download_data.py --max-lines 200000  # 疎通用の一部
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

REPO_ID = "BabyLM-community/BabyLM-2026-Strict"
SPLIT = "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--out-dir", default="data/text")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="取得する最大行数（疎通用。未指定なら全量）")
    parser.add_argument("--date", default=None,
                        help="取得日(YYYY-MM-DD)。未指定ならUTC現在日（再現用に記録）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "train.txt"

    streaming = args.max_lines is not None
    ds = load_dataset(args.repo_id, split=SPLIT, streaming=streaming)

    n_lines = 0
    n_words = 0
    with out_file.open("w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            f.write(text + "\n")
            n_lines += 1
            n_words += len(text.split())
            if args.max_lines is not None and n_lines >= args.max_lines:
                break

    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    source = {
        "repo_id": args.repo_id,
        "split": SPLIT,
        "download_date": date,
        "n_lines": n_lines,
        "n_words_whitespace": n_words,
        "max_lines": args.max_lines,
    }
    (out_dir / "SOURCE.json").write_text(json.dumps(source, indent=2, ensure_ascii=False))
    print(f"wrote {n_lines} lines / ~{n_words} words to {out_file}")
    print(f"provenance: {out_dir / 'SOURCE.json'}")


if __name__ == "__main__":
    main()
