"""features-off 対照（ablate_zero_vision）の不変条件。

- フラグ on: caption バッチの vision_features が全ゼロ（形状・他キーは不変）
- フラグ off（既定）: 特徴がそのまま通る（既存セル②④の挙動を変えない）
"""

import numpy as np
import torch

from babyloop.data.dataloader import MultimodalDataModule


def _make_module(ablate: bool) -> MultimodalDataModule:
    # setup() を経由せず collate 単体を検証（実データ不要）
    m = MultimodalDataModule.__new__(MultimodalDataModule)
    m.n_visual_tokens = 1
    m.pad_token_id = 0
    m.ablate_zero_vision = ablate
    m.feats = np.ones((8, 1, 4), dtype=np.float16)  # (M, stored, dim) 全1
    return m


def _batch():
    # (ids, vision_row, words) — _CaptionRecords の1要素と同型
    return [
        (torch.tensor([5, 6, 7]), 0, 3),
        (torch.tensor([8, 9]), 1, 2),
    ]


def test_zero_vision_on():
    out = _make_module(ablate=True)._collate_captions(_batch())
    assert torch.all(out["vision_features"] == 0)
    assert out["vision_features"].shape == (2, 1, 4)
    assert out["input_ids"].shape == (2, 3)  # pad 済み
    assert out["labels"][1, 2] == -100  # pad 位置は loss 除外のまま


def test_zero_vision_off_default():
    out = _make_module(ablate=False)._collate_captions(_batch())
    assert torch.all(out["vision_features"] == 1.0)
