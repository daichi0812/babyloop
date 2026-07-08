"""視覚特徴を我々の memmap 形式（feats.npy [M, V, dim] fp16 ＋ index.json）へ取り込む。

学習・eval に DINOv2 を載せず、視覚特徴を固定データ資産として供給するための前処理（ADR-0005）。

2モード（configs/data/multimodal.yaml の feature_source）:
  official … OSF(ad7qg)/multimodal_data の事前計算済み DINOv2 ViT-Base 特徴を取り込む（②の主経路。
             DINOv2 推論なし）。特徴は **V=1, 768次元 fp32**（単一 global/CLS ベクトル。OSF 実ファイル
             サイズで確認済）→ (N, 1, 768) に reshape して保存。
  images  … 配布 raw 画像から VisionFeatureExtractor（frozen DINOv2）で自前抽出（ablation/④。64 パッチ
             grid。data.dataloader._pool_visual と同じプール規則）。

出力: feature_dir/feats.npy（(M, V, dim) fp16）＋ feature_dir/index.json（{"rows": {image_id: row}}）。
MultimodalPreprocessor が index.json で caption→vision_row を解決する（image_id=源_行index で対応）。

使い方:
    uv run python scripts/precompute_vision.py --feature-source official --official-dir data/mm/osf
    uv run --group vision python scripts/precompute_vision.py --feature-source images --max-images 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _save_store(feature_dir: Path, feats: np.ndarray, image_ids: list[str]) -> None:
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(feature_dir / "feats.npy", feats.astype(np.float16))
    (feature_dir / "index.json").write_text(json.dumps({
        "rows": {iid: i for i, iid in enumerate(image_ids)},
        "meta": {"n": len(image_ids), "shape": list(feats.shape[1:])},
    }, indent=2))
    print(f"wrote feats {feats.shape} -> {feature_dir}/feats.npy (+ index.json)")


def _from_images(args) -> None:
    """配布 raw 画像 → frozen DINOv2 自前抽出（系統2）。"""
    from PIL import Image

    from babyloop.data.vision_features import VisionFeatureExtractor

    extractor = VisionFeatureExtractor(model_name=args.dinov2_name, n_tokens=args.stored_tokens)
    image_dir = Path(args.image_dir)
    paths = sorted(image_dir.glob("*.jpg"))
    if args.max_images:
        paths = paths[: args.max_images]
    feats, ids = [], []
    batch = 64
    for i in range(0, len(paths), batch):
        chunk = paths[i : i + batch]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        out = extractor.encode(imgs)
        feats.append(out["patches"].numpy())
        ids.extend(p.stem for p in chunk)
    _save_store(Path(args.feature_dir), np.concatenate(feats), ids)


def _from_official(args) -> None:
    """OSF 同梱の事前計算済み DINOv2 特徴 → 我々の memmap（②の主経路, ADR-0005）。

    OSF プロジェクト ad7qg の multimodal_data/ にある .npy（**V=1, 768次元 fp32**＝CLS/global
    ベクトル。OSF 実ファイルサイズで確認済）を読み、(N, 1, 768) fp16 memmap ＋ index.json にする。
    DINOv2 推論なし。image_id は ``源_行index``（download_mm_data の captions.jsonl と index 対応）。

      local_narr_dino_v2_states.npy            → ln_0, ln_1, ...
      cc_3M_dino_v2_states_1of2.npy + _2of2    → cc_0, cc_1, ...（2分割を連番で連結）
    """
    src = Path(args.official_dir)
    all_sources = {
        "ln": [src / "local_narr_dino_v2_states.npy"],
        "cc": [src / "cc_3M_dino_v2_states_1of2.npy", src / "cc_3M_dino_v2_states_2of2.npy"],
    }
    want = [s.strip() for s in args.sources.split(",") if s.strip()]
    cap = args.max_rows  # 源ごとの行上限（疎通/signal ラン用。None=全部）

    # 1パス目: 源ごとに取得行数（cap 適用）と次元（V=1, 768）を決める。
    plan, dim, total = [], None, 0
    for prefix in want:
        remaining = cap
        files = []
        for p in all_sources[prefix]:
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} が無い。先に scripts/download_mm_data.py で OSF(ad7qg)/multimodal_data を取得する。"
                )
            a = np.load(p, mmap_mode="r")
            assert a.ndim == 2, f"{p.name}: 想定は (N,768) の V=1。実 shape={a.shape}（grid なら別途プール要）"
            dim = a.shape[-1]
            n = a.shape[0] if remaining is None else min(a.shape[0], remaining)
            files.append((p, n))
            total += n
            if remaining is not None:
                remaining -= n
                if remaining <= 0:
                    break
        plan.append((prefix, files))

    # 2パス目: 出力 memmap へ書き込み（メモリ節約のため mmap→fp16 でコピー）。source 内は連番。
    out = np.empty((total, 1, dim), dtype=np.float16)
    ids, row = [], 0
    for prefix, files in plan:
        off = 0
        for p, n in files:
            a = np.load(p, mmap_mode="r")[:n].reshape(n, 1, dim)
            out[row : row + n] = a.astype(np.float16)
            ids.extend(f"{prefix}_{off + i}" for i in range(n))
            row += n
            off += n
    _save_store(Path(args.feature_dir), out, ids)
    print(f"[precompute:official] {total} rows ({[(p,len(f)) for p,f in plan]}) → {args.feature_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-source", choices=["official", "images"], default="official")
    p.add_argument("--feature-dir", default="data/mm/vision")
    p.add_argument("--official-dir", default="data/mm/osf",
                   help="OSF(ad7qg)/multimodal_data の .npy を置いたディレクトリ（official モード）")
    p.add_argument("--sources", default="ln,cc", help="official: 取り込む源（ln,cc のカンマ区切り）")
    p.add_argument("--max-rows", type=int, default=None, help="official: 源ごとの行上限（疎通/signal 用）")
    p.add_argument("--image-dir", default="data/mm/images")
    p.add_argument("--dinov2-name", default="facebook/dinov2-base")
    p.add_argument("--stored-tokens", type=int, default=64)
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()
    (_from_images if args.feature_source == "images" else _from_official)(args)


if __name__ == "__main__":
    main()
