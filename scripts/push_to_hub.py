"""学習済み checkpoint 群を HF Hub に提出形で push する（全セルで使い回す）。

`outputs/<run>/seed_<S>/` の `chck_*M`(28点) を**同名ブランチ**、`ckpt_final` を
`main` に push する。公式 `collate_preds.py` が revision=`chck_*M` を仮定するため
命名はそのまま使う。各 ckpt dir は modeling/config(auto_map)+tokenizer 同梱なので
`AutoModelForCausalLM(trust_remote_code=True)` で読める。

事前に認証（どちらか）: `uv run huggingface-cli login` / `export HF_TOKEN=hf_xxx`（write token）。
使い方:
    uv run python scripts/push_to_hub.py --repo-id <user>/babyloop-std-text \
        --run-dir outputs/std_text/seed_42
    # 確認だけ（push しない）:
    uv run python scripts/push_to_hub.py --repo-id <user>/babyloop-std-text \
        --run-dir outputs/std_text/seed_42 --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_UNIT = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def _words(name: str) -> int:
    m = re.match(r"chck_(\d+)([KMB])$", name)
    return int(m.group(1)) * _UNIT[m.group(2)] if m else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="例: <user>/babyloop-std-text")
    ap.add_argument("--run-dir", required=True, help="outputs/<run>/seed_<S>")
    ap.add_argument("--final-dir", default="ckpt_final", help="main に push する最終モデル dir 名")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="push せず対象を列挙するだけ")
    args = ap.parse_args()

    run = Path(args.run_dir)
    if not run.is_dir():
        raise SystemExit(f"run-dir が無い: {run}")

    final = run / args.final_dir
    chcks = sorted((d for d in run.glob("chck_*") if d.is_dir()), key=lambda p: _words(p.name))
    print(f"repo={args.repo_id}  final={'有' if final.is_dir() else '無'}  chck={len(chcks)}点")
    for d in chcks:
        print(f"  {d.name} -> branch {d.name}")

    if args.dry_run:
        print("[dry-run] push しません。")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    # 1) 最終モデル -> main
    if final.is_dir():
        api.upload_folder(folder_path=str(final), repo_id=args.repo_id,
                          commit_message="final model (main)")
        print(f"pushed {final.name} -> main")

    # 2) 中間 checkpoint -> 各ブランチ（語数の小さい順）
    for d in chcks:
        rev = d.name  # chck_7M
        api.create_branch(args.repo_id, branch=rev, exist_ok=True)
        api.upload_folder(folder_path=str(d), repo_id=args.repo_id, revision=rev,
                          commit_message=f"checkpoint {rev}")
        print(f"pushed {rev} -> branch {rev}")

    print("\n完了。提出評価は cwd=third_party/evaluation-pipeline/strict で（全て --group eval 必須）:")
    print("  # 事前: unzip -o -P BabyLM2025 evaluation_data/fast_eval/ewok_fast.zip  （EWoK fast 解凍）")
    print("  PROJ=<repo root>")
    print(f"  uv run --project $PROJ --group eval bash scripts/eval_zero_shot_fast_all_revisions.sh {args.repo_id} causal strict")
    print(f"  uv run --project $PROJ --group eval bash scripts/eval_zero_shot.sh {args.repo_id} causal")
    print(f"  uv run --project $PROJ --group eval bash scripts/eval_finetuning.sh {args.repo_id}")
    print(f"  uv run --project $PROJ --group eval bash scripts/collate_preds.sh {args.repo_id} causal strict")


if __name__ == "__main__":
    main()
