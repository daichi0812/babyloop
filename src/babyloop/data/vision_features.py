"""frozen 視覚エンコーダ（DINOv2）— 視覚特徴を**事前計算**するための data 側コンポーネント。

ADR-0002（frozen DINOv2・交絡制御）＋ ADR-0005（オフライン事前計算）。frozen encoder は
勾配を流さず word budget 外で、`scripts/precompute_vision.py` から一度だけ回して特徴を
`.npy`/memmap にキャッシュする。学習ループ・eval は特徴を読むだけ。

**`models/modeling_babyloop.py`（公式 evaluation-pipeline が torch/transformers のみで
trust_remote_code import する自己完結ファイル）には入れない**。ここは前処理専用なので
DINOv2 や画像処理（PIL）に依存してよい。
"""

from __future__ import annotations

import math

import torch


class VisionFeatureExtractor:
    """frozen DINOv2 で画像→視覚特徴（パッチをプール＋CLS）を抽出する。

    Args:
        model_name: HF のモデル ID。既定 ``facebook/dinov2-base``（ViT-B/14, hidden=768=d_model）。
            HF キャッシュ（``~/.cache/huggingface``）に落ちる（torch.hub は使わない=cache 分散回避）。
        n_tokens: パッチを 2D 平均プールする目標トークン数（平方数。既定 64=8×8）。
            学習時トークン圧縮の上限（学習時にここからさらにプールできる、ADR-0005）。
        layers: 取り出す中間層インデックス（④の階層注入用）。``None`` で最終層のみ（②）。
        device: 推論デバイス。未指定なら cuda→cpu。
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        n_tokens: int = 64,
        layers: list[int] | None = None,
        device: str | None = None,
    ):
        from transformers import AutoImageProcessor, AutoModel

        self.model_name = model_name
        self.n_tokens = n_tokens
        self.layers = layers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)  # frozen: 予算外・勾配を流さない（ADR-0002）
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.feature_dim = int(self.model.config.hidden_size)

    def _pool(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, P, H) のパッチ列を 2D 平均プールで (B, n_tokens, H) に圧縮する。"""
        B, P, H = patches.shape
        g = int(round(math.sqrt(P)))
        if g * g != P:
            raise ValueError(f"パッチ数 {P} が平方数でない（プール不能）")
        tg = int(round(math.sqrt(self.n_tokens)))
        if tg * tg != self.n_tokens:
            raise ValueError(f"n_tokens={self.n_tokens} が平方数でない")
        x = patches.transpose(1, 2).reshape(B, H, g, g)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (tg, tg))
        return x.reshape(B, H, tg * tg).transpose(1, 2).contiguous()

    @torch.no_grad()
    def encode(self, images) -> dict[str, torch.Tensor]:
        """画像バッチ（PIL.Image のリスト等）→ 視覚特徴。

        Returns:
            ``{"patches": (B, n_tokens, H) fp16, "cls": (B, H) fp16}``。
            ``layers`` 指定時は ``"layers": (B, n_layers, n_tokens, H)`` も付く（④）。
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=self.layers is not None)
        hs = out.last_hidden_state  # (B, 1+P, H): index0=CLS, 残り=patch tokens
        result = {
            "patches": self._pool(hs[:, 1:]).to(torch.float16).cpu(),
            "cls": hs[:, 0].to(torch.float16).cpu(),
        }
        if self.layers is not None:
            # hidden_states は embedding 出力 + 各層出力。負/正どちらの index も許容。
            stacked = torch.stack(
                [self._pool(out.hidden_states[l][:, 1:]) for l in self.layers], dim=1
            )
            result["layers"] = stacked.to(torch.float16).cpu()
        return result
