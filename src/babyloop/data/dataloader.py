"""学習用dataloader。

text-only（①③）は packed causal LM。multimodal（②④）は text packed blocks と
画像-caption 単位レコードを **homogeneous バッチ**（全-text or 全-caption）で混ぜる
（ragged な視覚バッチを避ける）。視覚特徴は事前計算済み memmap から collate 時に gather
し、学習時に保存トークン(64)→`n_visual_tokens` へ再プールする（ADR-0005）。
"""

from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from babyloop.data.preprocessing import mm_caption_dir, mm_text_dir, tokenized_dir


class BaseDataModule(ABC):
    """データセット構築とdataloader提供の共通インターフェース。

    Trainer はモダリティを意識せず、このインターフェースだけに依存する。
    """

    def __init__(self, data_cfg, train_cfg):
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg

    @abstractmethod
    def setup(self) -> None:
        """前処理済みデータの読み込みとデータセット構築。"""

    @abstractmethod
    def train_dataloader(self):
        """学習用dataloaderを返す。"""


class _PackedBlocks(Dataset):
    """(n_blocks, seq_len) の packed トークン列。各ブロックが1サンプル。

    各ブロックの**実語数**（A-2）も同梱し、消費ブロックの語数を正確に積算できる
    ようにする（words_seen の精密駆動）。block_words が無い旧データは比率で近似。
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, block_words: np.ndarray | None = None,
                 words_per_token: float = 0.0):
        n_blocks = len(tokens) // seq_len
        self.blocks = tokens[: n_blocks * seq_len].reshape(n_blocks, seq_len)
        if block_words is not None and len(block_words) >= n_blocks:
            self.block_words = block_words[:n_blocks].astype(np.int64)
        else:
            self.block_words = np.full(n_blocks, round(seq_len * words_per_token), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        # uint16 は torch 非対応のため int64 へ。
        return torch.from_numpy(self.blocks[idx].astype(np.int64)), int(self.block_words[idx])


class TextDataModule(BaseDataModule):
    """テキストのみのDataModule（packed causal LM）。"""

    def setup(self) -> None:
        out_dir = tokenized_dir(self.data_cfg)
        meta_path = out_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} が無い。先に scripts/preprocess.py を実行する。"
            )
        self.meta = json.loads(meta_path.read_text())
        tokens = np.load(out_dir / "tokens.npy")
        bw_path = out_dir / "block_words.npy"
        block_words = np.load(bw_path) if bw_path.exists() else None
        self.dataset = _PackedBlocks(
            tokens, self.meta["max_seq_len"], block_words=block_words,
            words_per_token=self.words_per_token,
        )

    @property
    def words_per_token(self) -> float:
        return float(self.meta.get("words_per_token", 0.0))

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=int(self.train_cfg.batch_size),
            shuffle=True,
            drop_last=True,
        )


# --- multimodal（②④）-------------------------------------------------------


class _CaptionRecords(Dataset):
    """画像-caption 単位レコード。1サンプル = (caption_ids, vision_row, n_words)。"""

    def __init__(self, flat_tokens: np.ndarray, offsets: np.ndarray,
                 vision_rows: np.ndarray, words: np.ndarray):
        self.flat = flat_tokens
        self.offsets = offsets
        self.vision_rows = vision_rows
        self.words = words

    def __len__(self) -> int:
        return len(self.vision_rows)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int, int]:
        s, e = int(self.offsets[idx]), int(self.offsets[idx + 1])
        ids = torch.from_numpy(self.flat[s:e].astype(np.int64))
        return ids, int(self.vision_rows[idx]), int(self.words[idx])


def _pool_visual(x: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """(B, S, H) の視覚トークンを 2D 平均プールで (B, target, H) に再圧縮する。"""
    B, S, H = x.shape
    if target_tokens <= 0 or target_tokens == S:
        return x
    g = int(round(math.sqrt(S)))
    tg = int(round(math.sqrt(target_tokens)))
    if g * g != S or tg * tg != target_tokens:
        raise ValueError(f"視覚プールは平方数前提（S={S}, target={target_tokens}）")
    x = x.transpose(1, 2).reshape(B, H, g, g)
    x = F.adaptive_avg_pool2d(x, (tg, tg))
    return x.reshape(B, H, tg * tg).transpose(1, 2).contiguous()


class _MixedLoader:
    """text/caption の2 loader を ``p_text`` 確率で homogeneous に混ぜる無限イテレータ。

    どちらかが尽きたら再イテレート（cycle）。学習は Trainer の words/steps 上限で停止する。
    片方が空なら他方のみを引く。
    """

    def __init__(self, text_loader, caption_loader, p_text: float, seed: int):
        self.text_loader = text_loader
        self.caption_loader = caption_loader
        self.seed = seed
        has_text = text_loader is not None
        has_cap = caption_loader is not None
        if not has_text and not has_cap:
            raise ValueError("text/caption とも空：MM データが無い")
        self.p_text = 1.0 if not has_cap else (0.0 if not has_text else p_text)

    def __iter__(self):
        rng = random.Random(self.seed)
        ti = iter(self.text_loader) if self.text_loader is not None else None
        ci = iter(self.caption_loader) if self.caption_loader is not None else None
        while True:
            pick_text = rng.random() < self.p_text
            if pick_text:
                try:
                    yield next(ti)
                except StopIteration:
                    ti = iter(self.text_loader)
                    yield next(ti)
            else:
                try:
                    yield next(ci)
                except StopIteration:
                    ci = iter(self.caption_loader)
                    yield next(ci)


class MultimodalDataModule(BaseDataModule):
    """テキスト＋画像-caption 対のDataModule（②④）。

    Args:
        n_visual_tokens: 学習時に視覚特徴をプールする目標トークン数（0 で保存トークンのまま）。
        pad_token_id: caption の pad に使う id（padは labels で -100）。
        ablate_zero_vision: True で視覚特徴をゼロ化（features-off 対照。データ組成・prefix
            機構・budget 分割は同一のまま「視覚情報」だけを消す）。
    """

    def __init__(self, data_cfg, train_cfg, n_visual_tokens: int = 0, pad_token_id: int = 0,
                 ablate_zero_vision: bool = False):
        super().__init__(data_cfg, train_cfg)
        self.n_visual_tokens = int(n_visual_tokens)
        self.pad_token_id = int(pad_token_id)
        self.ablate_zero_vision = bool(ablate_zero_vision)

    def setup(self) -> None:
        # text 部分
        td = mm_text_dir(self.data_cfg)
        tmeta = td / "meta.json"
        if not tmeta.exists():
            raise FileNotFoundError(f"{tmeta} が無い。先に scripts/preprocess.py (mm) を実行する。")
        self.text_meta = json.loads(tmeta.read_text())
        text_tokens = np.load(td / "tokens.npy")
        bw = td / "block_words.npy"
        self.text_ds = _PackedBlocks(
            text_tokens, self.text_meta["max_seq_len"],
            block_words=np.load(bw) if bw.exists() else None,
            words_per_token=float(self.text_meta.get("words_per_token", 0.0)),
        ) if len(text_tokens) else None

        # caption 部分
        cd = mm_caption_dir(self.data_cfg)
        flat = np.load(cd / "caption_tokens.npy")
        offsets = np.load(cd / "caption_offsets.npy")
        vrows = np.load(cd / "caption_vision_rows.npy")
        cwords = np.load(cd / "caption_words.npy")
        self.caption_ds = _CaptionRecords(flat, offsets, vrows, cwords) if len(vrows) else None

        # 事前計算済み視覚特徴（memmap, fp16）。(M, stored_tokens, feature_dim)
        feats_path = self.data_cfg.feature_dir
        self.feats = np.load(f"{feats_path}/feats.npy", mmap_mode="r")

    def _collate_text(self, batch):
        input_ids = torch.stack([b[0] for b in batch])
        words = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids, "batch_words": words}

    def _collate_captions(self, batch):
        ids_list = [b[0] for b in batch]
        rows = [b[1] for b in batch]
        words = torch.tensor([b[2] for b in batch], dtype=torch.long)
        L = max(int(x.numel()) for x in ids_list)
        B = len(batch)
        input_ids = torch.full((B, L), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            n = int(ids.numel())
            input_ids[i, :n] = ids
            attention_mask[i, :n] = 1
            labels[i, :n] = ids  # pad 位置は -100 のまま（loss から除外）
        feats = np.stack([np.asarray(self.feats[r]) for r in rows])  # (B, stored, dim) fp16
        vision_features = _pool_visual(
            torch.from_numpy(feats).float(), self.n_visual_tokens
        )
        if self.ablate_zero_vision:
            vision_features = torch.zeros_like(vision_features)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "vision_features": vision_features,
            "batch_words": words,
        }

    def train_dataloader(self):
        bs = int(self.train_cfg.batch_size)
        text_loader = (
            DataLoader(self.text_ds, batch_size=bs, shuffle=True, drop_last=True,
                       collate_fn=self._collate_text)
            if self.text_ds is not None else None
        )
        caption_loader = (
            DataLoader(self.caption_ds, batch_size=bs, shuffle=True, drop_last=False,
                       collate_fn=self._collate_captions)
            if self.caption_ds is not None else None
        )
        return _MixedLoader(
            text_loader, caption_loader,
            p_text=float(self.data_cfg.text_caption_ratio),
            seed=int(self.train_cfg.seed),
        )
