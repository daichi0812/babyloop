"""モデル実装。LoopedTransformer（K=1で標準に縮退）と視覚融合 connector。

視覚エンコーダ（frozen DINOv2）は data 側（``data/vision_features.py``）に置き、特徴を
オフライン事前計算する（ADR-0005）。modeling には torch 純正の ``VisualConnector`` のみ
（自己完結維持）。融合の注入場所は ``config.visual_inject_iters`` が決める（ADR-0006）。
"""

from babyloop.models.configuration_babyloop import BabyloopConfig
from babyloop.models.looped_transformer import (
    LoopedForCausalLM,
    LoopedModel,
    LoopedTransformer,
)
from babyloop.models.modeling_babyloop import VisualConnector

__all__ = [
    "BabyloopConfig",
    "LoopedForCausalLM",
    "LoopedModel",
    "LoopedTransformer",
    "VisualConnector",
]
