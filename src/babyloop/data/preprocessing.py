"""コーパス前処理。scripts/preprocess.py から呼ばれる。

生コーパス → BPEトークナイザ学習 → トークナイズ＆packing → 保存。語数は
WordBudgetTracker で計上し、固定word budget（①は100M）を超える分は打ち切る。

②④（multimodal）はテキストに加えて画像-キャプション対を扱う。キャプション語数も
word budget に計上し、`text_caption_ratio` で text/caption の予算を分ける。視覚特徴は
別途 `scripts/precompute_vision.py` で frozen DINOv2 から事前計算済み（ADR-0005）で、
ここでは caption→vision_row の対応付けだけ行う。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from babyloop.data.tokenizer import train_bpe_tokenizer
from babyloop.data.word_budget import WordBudgetTracker


def tokenized_dir(data_cfg) -> Path:
    """前処理済みトークン列の保存先（TextDataModule と共有）。"""
    return Path(data_cfg.text_dir) / "tokenized"


def mm_text_dir(data_cfg) -> Path:
    """multimodal: テキスト部分（packed blocks）の保存先。"""
    return Path(data_cfg.processed_dir) / "text"


def mm_caption_dir(data_cfg) -> Path:
    """multimodal: キャプション部分（画像-caption 単位レコード）の保存先。"""
    return Path(data_cfg.processed_dir) / "captions"


def pack_lines_to_blocks(tokenizer, lines: list[str], max_seq_len: int, vocab_size: int):
    """行リストをトークナイズ → 連結 → ``max_seq_len`` で packing する。

    各ブロックの実語数（A-2）も返す: 行の最終トークンが属するブロックへ行の全語数を計上
    （残余に落ちた行は計上しない＝トークンを捨てるのと整合）。TextPreprocessor と
    MultimodalPreprocessor のテキスト部分で共有する（packing ロジックの単一定義）。

    Returns: ``(tokens, block_words, packed_words)``。
    """
    dtype = np.uint16 if vocab_size < 2**16 else np.uint32
    per_line_tokens: list[np.ndarray] = []
    per_line_words: list[int] = []
    batch = 4096
    for i in range(0, len(lines), batch):
        enc = tokenizer(lines[i : i + batch])["input_ids"]
        for j, ids in enumerate(enc):
            per_line_tokens.append(np.asarray(ids, dtype=dtype))
            per_line_words.append(WordBudgetTracker.count_words(lines[i + j]))
    tokens = np.concatenate(per_line_tokens) if per_line_tokens else np.zeros(0, dtype=dtype)

    n_blocks = len(tokens) // max_seq_len
    tokens = tokens[: n_blocks * max_seq_len]

    block_words = np.zeros(n_blocks, dtype=np.int64)
    cum = 0
    for arr, w in zip(per_line_tokens, per_line_words):
        cum += len(arr)
        bidx = (cum - 1) // max_seq_len
        if bidx < n_blocks:
            block_words[bidx] += w
    return tokens, block_words, int(block_words.sum())


class BasePreprocessor(ABC):
    """前処理の共通インターフェース。

    生コーパス → トークナイズ済みデータセットへの変換と、
    WordBudgetTracker による語数計上をここで行う。

    Args:
        cfg: Hydra合成済みの全体config（data と model を参照する）。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_cfg = cfg.data

    @abstractmethod
    def run(self) -> None:
        """前処理を実行し、結果を data_cfg のパスへ書き出す。"""


class TextPreprocessor(BasePreprocessor):
    """テキストのみコーパス（configs/data/text_only.yaml）の前処理。"""

    def _iter_budgeted_lines(self) -> list[str]:
        """text_dir のテキストを word budget まで読み込む（超過分は打ち切り）。"""
        tracker = WordBudgetTracker(budget=int(self.data_cfg.word_budget))
        text_dir = Path(self.data_cfg.text_dir)
        paths = sorted(p for p in text_dir.glob("*.txt"))
        if not paths:
            raise FileNotFoundError(
                f"{text_dir} に .txt が無い。先に scripts/download_data.py を実行する。"
            )
        lines: list[str] = []
        for path in paths:
            with path.open(encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    n = WordBudgetTracker.count_words(line)
                    if n > tracker.remaining:
                        self.consumed_words = tracker.consumed
                        return lines  # 予算到達で打ち切り
                    tracker.add(n)
                    lines.append(line)
        self.consumed_words = tracker.consumed
        return lines

    def run(self) -> None:
        vocab_size = int(self.cfg.model.vocab_size)
        max_seq_len = int(self.cfg.model.max_seq_len)

        lines = self._iter_budgeted_lines()
        print(f"[preprocess] {len(lines)} lines, {self.consumed_words} words (budget kept)")

        # BPE学習 → tokenizer_path へ保存。
        tokenizer = train_bpe_tokenizer(
            iter(lines), vocab_size=vocab_size,
            save_dir=str(self.data_cfg.tokenizer_path), max_seq_len=max_seq_len,
        )

        tokens, block_words, packed_words = pack_lines_to_blocks(
            tokenizer, lines, max_seq_len, vocab_size
        )

        out_dir = tokenized_dir(self.data_cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "tokens.npy", tokens)
        np.save(out_dir / "block_words.npy", block_words)
        meta = {
            "total_tokens": int(len(tokens)),
            "total_words": int(self.consumed_words),
            "packed_words": int(packed_words),
            "max_seq_len": max_seq_len,
            "vocab_size": vocab_size,
            "n_blocks": int(len(tokens) // max_seq_len),
            "words_per_token": (self.consumed_words / len(tokens)) if len(tokens) else 0.0,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[preprocess] packed {meta['n_blocks']} blocks x {max_seq_len} -> {out_dir} | meta={meta}")


class MultimodalPreprocessor(BasePreprocessor):
    """画像-キャプション対を含むコーパス（configs/data/multimodal.yaml）の前処理。

    word budget を ``text_caption_ratio`` で text/caption に分割し、双方の語数を
    WordBudgetTracker で計上する（キャプション語数も100M予算に乗る）。テキスト部分は
    ①と同じ packing。キャプション部分は画像-caption 単位レコード（視覚 prefix が特定
    caption に束縛されるため packing しない）。視覚特徴は事前計算済み（``feature_dir``）で、
    ここでは caption の image_id を index.json で vision_row に解決する（特徴が無い＝
    リンク腐敗で落ちた画像の caption はスキップ）。
    """

    def _load_or_train_tokenizer(self, text_lines: list[str], vocab_size: int, max_seq_len: int):
        """①の凍結 tokenizer を再利用（BLiMP 比較性の前提）。無ければ text で学習（疎通用）。"""
        from babyloop.data.tokenizer import load_tokenizer

        path = str(self.data_cfg.tokenizer_path)
        if (Path(path) / "tokenizer.json").exists():
            print(f"[preprocess] reuse frozen tokenizer: {path}")
            return load_tokenizer(path)
        print(f"[preprocess] tokenizer 不在 → text で学習（疎通用）: {path}")
        return train_bpe_tokenizer(
            iter(text_lines), vocab_size=vocab_size, save_dir=path, max_seq_len=max_seq_len
        )

    def _iter_budgeted_text(self, budget: int) -> tuple[list[str], int]:
        """text_dir のテキストを text 予算まで読み込む。"""
        tracker = WordBudgetTracker(budget=budget)
        text_dir = Path(self.data_cfg.text_dir)
        paths = sorted(p for p in text_dir.glob("*.txt"))
        if not paths:
            raise FileNotFoundError(
                f"{text_dir} に .txt が無い。先に scripts/download_data.py を実行する。"
            )
        lines: list[str] = []
        for path in paths:
            with path.open(encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    n = WordBudgetTracker.count_words(line)
                    if n > tracker.remaining:
                        return lines, tracker.consumed
                    tracker.add(n)
                    lines.append(line)
        return lines, tracker.consumed

    def _vision_index(self) -> dict[str, int]:
        """feature_dir/index.json から image_id -> vision_row の対応を読む。"""
        idx_path = Path(self.data_cfg.feature_dir) / "index.json"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"{idx_path} が無い。先に scripts/precompute_vision.py を実行する（ADR-0005）。"
            )
        data = json.loads(idx_path.read_text())
        return data["rows"] if "rows" in data else data  # meta 付き/無しの両対応

    def run(self) -> None:
        vocab_size = int(self.cfg.model.vocab_size)
        max_seq_len = int(self.cfg.model.max_seq_len)
        dc = self.data_cfg
        ratio = float(dc.text_caption_ratio)
        budget = int(dc.word_budget)
        text_budget = round(budget * ratio)
        caption_budget = budget - text_budget

        # --- tokenizer（①再利用 or 疎通学習）---
        text_lines, text_words = self._iter_budgeted_text(text_budget)
        tokenizer = self._load_or_train_tokenizer(text_lines, vocab_size, max_seq_len)

        # --- text 部分（①と同じ packing）---
        tokens, block_words, packed_words = pack_lines_to_blocks(
            tokenizer, text_lines, max_seq_len, vocab_size
        )
        text_out = mm_text_dir(dc)
        text_out.mkdir(parents=True, exist_ok=True)
        np.save(text_out / "tokens.npy", tokens)
        np.save(text_out / "block_words.npy", block_words)
        n_blocks = int(len(tokens) // max_seq_len)
        (text_out / "meta.json").write_text(json.dumps({
            "total_tokens": int(len(tokens)), "total_words": int(text_words),
            "packed_words": int(packed_words), "max_seq_len": max_seq_len,
            "vocab_size": vocab_size, "n_blocks": n_blocks,
            "words_per_token": (text_words / len(tokens)) if len(tokens) else 0.0,
        }, indent=2))

        # --- caption 部分（画像-caption 単位。事前計算済み視覚特徴へ row 解決）---
        rows = self._vision_index()
        dtype = np.uint16 if vocab_size < 2**16 else np.uint32
        backfill = bool(dc.get("backfill_ln", False))  # CC3M 不足時 LN 補填（既定 false=締める）
        # 長文 caption（特に LN ナラティブ）が prefix(V)+max_seq_len を超えないよう token 列を
        # **tail-truncate**（先頭=主要描写を残す。drop でなくデータを保つ。ADR-0005 既定）。
        # budget は語数（whitespace）で計上＝truncate で切れた分も語数は数える（≤budget 側に保守的）。
        n_visual = int(self.cfg.model.get("vision_encoder", {}).get("n_visual_tokens", 0))
        cap_max_len = max(1, max_seq_len - n_visual)
        tracker = WordBudgetTracker(budget=caption_budget)
        flat: list[np.ndarray] = []
        offsets: list[int] = [0]
        vision_rows: list[int] = []
        cap_words: list[int] = []
        n_skipped_no_feat = 0
        n_truncated = 0
        cap_path = Path(dc.caption_dir) / "captions.jsonl"
        if not cap_path.exists():
            raise FileNotFoundError(
                f"{cap_path} が無い。先に scripts/download_mm_data.py を実行する。"
            )
        with cap_path.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                iid, cap = str(rec["image_id"]), str(rec["caption"]).strip()
                if not cap:
                    continue
                if iid not in rows:
                    n_skipped_no_feat += 1  # 特徴が無い（リンク腐敗で落ちた画像）→スキップ
                    continue
                n = WordBudgetTracker.count_words(cap)
                if n > tracker.remaining:
                    break  # caption 予算到達で打ち切り（既定: 得られた語数で締める）
                tracker.add(n)
                ids = np.asarray(tokenizer(cap)["input_ids"], dtype=dtype)
                if len(ids) > cap_max_len:
                    ids = ids[:cap_max_len]  # tail-truncate（prefix V + caption ≤ max_seq_len）
                    n_truncated += 1
                flat.append(ids)
                offsets.append(offsets[-1] + len(ids))
                vision_rows.append(int(rows[iid]))
                cap_words.append(n)

        cap_out = mm_caption_dir(dc)
        cap_out.mkdir(parents=True, exist_ok=True)
        cap_tokens = np.concatenate(flat) if flat else np.zeros(0, dtype=dtype)
        np.save(cap_out / "caption_tokens.npy", cap_tokens)
        np.save(cap_out / "caption_offsets.npy", np.asarray(offsets, dtype=np.int64))
        np.save(cap_out / "caption_vision_rows.npy", np.asarray(vision_rows, dtype=np.int64))
        np.save(cap_out / "caption_words.npy", np.asarray(cap_words, dtype=np.int64))
        (cap_out / "meta.json").write_text(json.dumps({
            "n_captions": len(vision_rows),
            "caption_words": int(tracker.consumed),
            "n_skipped_no_feature": n_skipped_no_feat,
            "n_truncated": n_truncated,
            "cap_max_len": cap_max_len,
            "vocab_size": vocab_size,
            "backfill_ln": backfill,
        }, indent=2))

        total_words = text_words + tracker.consumed
        print(
            f"[preprocess:mm] text={n_blocks} blocks ({text_words}w) / "
            f"captions={len(vision_rows)} ({tracker.consumed}w, skipped_no_feat={n_skipped_no_feat}) / "
            f"total={total_words}w (budget={budget}, text:caption={ratio}) -> {dc.processed_dir}"
        )
