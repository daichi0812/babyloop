"""BabyloopConfig — HF ``PretrainedConfig`` 互換のモデル設定。

このファイルは ``save_pretrained`` 時に checkpoint ディレクトリへ複製され、
公式 evaluation-pipeline 側（別プロセス）から ``trust_remote_code=True`` で
import される。そのため **transformers と標準ライブラリ以外に依存しない**
（``babyloop`` パッケージ内部を import しない）こと。

フィールドと configs/model/*.yaml の対応は docs/design.md §6 を参照。
"""

from __future__ import annotations

from transformers import PretrainedConfig


class BabyloopConfig(PretrainedConfig):
    """ループドTransformer（K=1で標準に縮退）の設定。

    軸A（標準 vs 再帰）は ``k`` のみで切り替える単一実装。視覚（②④）の口も
    持つが、テキストのみ（①③）では ``fusion=None`` で無効。
    """

    model_type = "babyloop"

    def __init__(
        self,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        ffn_hidden: int = 2048,
        n_prelude: int = 0,
        n_core: int = 12,
        n_coda: int = 0,
        k: int = 1,
        inject_input: bool = False,
        vocab_size: int = 16000,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
        rms_eps: float = 1e-5,
        tie_embeddings: bool = True,
        bias: bool = False,
        dropout: float = 0.0,
        fusion: str | None = None,
        vision_feature_dim: int = 768,
        n_visual_tokens: int = 0,
        connector_type: str = "mlp",
        visual_inject_iters: int = 0,
        visual_inject_mode: str = "prefix_refresh",
        vision_layers: list[int] | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_hidden = ffn_hidden
        self.n_prelude = n_prelude
        self.n_core = n_core
        self.n_coda = n_coda
        self.k = k
        self.inject_input = inject_input
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.rms_eps = rms_eps
        self.tie_embeddings = tie_embeddings
        self.bias = bias
        self.dropout = dropout
        # --- 視覚（②④。text では fusion=None で完全に無効）---
        # fusion: 視覚 on/off ゲート（None=テキスト / not-None=視覚あり）。projector 型は持たない（ADR-0006）。
        self.fusion = fusion
        self.vision_feature_dim = vision_feature_dim  # frozen encoder の特徴次元（ViT-B/14=768）
        self.n_visual_tokens = n_visual_tokens        # prefix する視覚トークン数（学習時圧縮後）
        self.connector_type = connector_type          # projector 型: mlp / identity（=naive 劣化）
        self.visual_inject_iters = visual_inject_iters # 注入場所 SoT: 0=prefix(②) / >0=staged(④)
        self.visual_inject_mode = visual_inject_mode   # ④注入形: prefix_refresh(B,主系) / broadcast(A)
        self.vision_layers = vision_layers             # ④の階層注入で使う中間層（②は未使用）

        # HF 汎用ユーティリティが参照するエイリアス。
        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_hidden_layers = n_layers
        self.max_position_embeddings = max_seq_len

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_embeddings,
            **kwargs,
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
