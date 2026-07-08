"""ループドTransformer本体（公開エイリアス）。

実体は HF ``PreTrainedModel`` 互換の :class:`LoopedForCausalLM`
（[modeling_babyloop.py](modeling_babyloop.py)）。2×2デザインの要因A
（標準 vs 再帰）は ``BabyloopConfig.k`` のみで切り替える単一実装で、
``k=1`` で標準Transformerに厳密に縮退する（``tests/test_k1_equivalence.py``）。
視覚特徴の融合（②④）は ``BabyloopConfig.fusion`` の config 駆動で、引数では持たない。
"""

from babyloop.models.configuration_babyloop import BabyloopConfig
from babyloop.models.modeling_babyloop import (
    LoopedCausalLMOutput,
    LoopedForCausalLM,
    LoopedModel,
    LoopedModelOutput,
)

# 設計ドキュメント・既存 import 互換のための別名。
LoopedTransformer = LoopedForCausalLM

__all__ = [
    "BabyloopConfig",
    "LoopedForCausalLM",
    "LoopedModel",
    "LoopedTransformer",
    "LoopedCausalLMOutput",
    "LoopedModelOutput",
]
