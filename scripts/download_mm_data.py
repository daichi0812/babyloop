"""BabyLM マルチモーダル（画像-caption）データの取得 — 配布元は **OSF プロジェクト ad7qg**。

HF ではなく OSF（osf.io/ad7qg）の ``multimodal_data/`` に caption と事前計算済み DINOv2 特徴がある:
  local_narr_captions.json / local_narr_dino_v2_states.npy        … Localized Narratives 27M語
  cc_3M_captions.json / cc_3M_dino_v2_states_{1,2}of2.npy          … Conceptual Captions 23M語
  train_50M.zip                                                    … text 50M（text_dir 用）
caption(json) と特徴(.npy) は **行 index で対応**（画像 ID 不要）。画像は再配布されない（COCO/Open
Images/CC3M のライセンス）ので、official モードでは画像 DL 不要＝OSF の特徴をそのまま使う（ADR-0005）。

このスクリプトの official モードの役割:
  1. OSF(ad7qg)/multimodal_data の .json/.npy を ``--osf-dir`` へ取得（osfclient 推奨。未導入なら手動DLを案内）。
  2. caption json → ``caption_dir/captions.jsonl``（{"image_id": "源_行index", "caption": str}）に変換。
  .npy は scripts/precompute_vision.py --feature-source official が ``--official-dir=<osf-dir>`` で取り込む。

⚠️ caption json の内部構造（list[str] か list[dict] か、dict のキー名）は OSF 実ファイル/README で要確認。
   下記 _extract_captions は list[str] / list[dict] を defensive に扱うが、確定後に調整すること。

使い方:
    pip/uv で osfclient を入れ: uv run --with osfclient python scripts/download_mm_data.py --osf-dir data/mm/osf
    （または手動DL後）   uv run python scripts/download_mm_data.py --skip-fetch --osf-dir data/mm/osf
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

OSF_PROJECT = "ad7qg"
# (源 prefix, caption json ファイル名)。.npy 側と同じ源・同じ行順で対応する。
CAPTION_FILES = [
    ("ln", "local_narr_captions.json"),
    ("cc", "cc_3M_captions.json"),
]


def _extract_captions(obj) -> list[str]:
    """caption json を caption 文字列のリストへ（行 index が .npy 行に対応）。

    ⚠️ 構造は OSF 実ファイルで要確認。list[str] / list[dict(caption|text|...)] を defensive に扱う。
    """
    if not isinstance(obj, list):
        raise ValueError(f"想定は list（行=.npy行）。実際は {type(obj)}。README で構造確認のこと。")
    out = []
    for item in obj:
        if isinstance(item, str):
            out.append(item.strip())
        elif isinstance(item, dict):
            cap = item.get("caption") or item.get("text") or item.get("annotation") or ""
            out.append(str(cap).strip())
        else:
            out.append("")
    return out


def _maybe_fetch_osf(osf_dir: Path) -> None:
    """osfclient で OSF(ad7qg)/multimodal_data を取得（未導入なら手動DLを案内）。"""
    try:
        from osfclient.api import OSF
    except ImportError:
        print(
            f"[download_mm] osfclient 未導入。手動で https://osf.io/{OSF_PROJECT}/files の "
            f"multimodal_data/ を {osf_dir} へ落とすか、`uv run --with osfclient ...` で再実行。"
        )
        return
    osf_dir.mkdir(parents=True, exist_ok=True)
    storage = OSF().project(OSF_PROJECT).storage("osfstorage")
    for f in storage.files:
        if "/multimodal_data/" not in f.path:
            continue
        dest = osf_dir / Path(f.path).name
        print(f"[download_mm] OSF → {dest} ({f.path})")
        with dest.open("wb") as fh:
            f.write_to(fh)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--osf-dir", default="data/mm/osf", help="OSF multimodal_data の .json/.npy 置き場")
    p.add_argument("--caption-dir", default="data/mm/captions")
    p.add_argument("--feature-source", choices=["official", "images"], default="official")
    p.add_argument("--skip-fetch", action="store_true", help="OSF 取得を飛ばす（手動DL済み）")
    p.add_argument("--max-pairs", type=int, default=None, help="源ごとの上限（疎通用）")
    p.add_argument("--date", default=None)
    args = p.parse_args()

    osf_dir = Path(args.osf_dir)
    if not args.skip_fetch and args.feature_source == "official":
        _maybe_fetch_osf(osf_dir)

    cap_dir = Path(args.caption_dir)
    cap_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    counts = {}
    with (cap_dir / "captions.jsonl").open("w", encoding="utf-8") as out:
        for prefix, fname in CAPTION_FILES:
            path = osf_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"{path} が無い。OSF(ad7qg)/multimodal_data から取得する。")
            captions = _extract_captions(json.loads(path.read_text(encoding="utf-8")))
            if args.max_pairs is not None:
                captions = captions[: args.max_pairs]
            for i, cap in enumerate(captions):
                if not cap:
                    continue
                out.write(json.dumps({"image_id": f"{prefix}_{i}", "caption": cap},
                                     ensure_ascii=False) + "\n")
            counts[prefix] = len(captions)
            total += len(captions)

    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (cap_dir / "SOURCE.json").write_text(json.dumps({
        "osf_project": OSF_PROJECT, "download_date": date, "n_pairs": total, "per_source": counts,
        "note": "image_id=源_行index で .npy(precompute) と対応。caption json 構造は README で要確認。",
    }, indent=2, ensure_ascii=False))
    print(f"wrote {total} caption pairs to {cap_dir}/captions.jsonl (per_source={counts})")


if __name__ == "__main__":
    main()
