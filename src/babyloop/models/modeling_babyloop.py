"""ループドTransformer本体（HF ``PreTrainedModel`` 互換・単一実装）。

設計 docs/design.md §2 の Llama 系レシピ（RMSNorm / RoPE / SwiGLU /
bias なし / weight tying）を素の PyTorch で実装。``forward`` は K 回ループする
形で書き、``k=1`` で標準Transformerに厳密に縮退する（``tests/test_k1_equivalence.py``）。

このファイルは ``save_pretrained`` 時に checkpoint へ複製され、公式
evaluation-pipeline（別プロセス・``trust_remote_code=True``）から import される。
そのため **torch / transformers / 標準ライブラリ以外に依存しない**こと。
probing 用のループ毎中間表現は HF 標準 ``hidden_states``（層ごと）と混ぜず、
別フィールド ``loop_hidden_states`` に格納する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_babyloop import BabyloopConfig


# --- ビルディングブロック（Llama系レシピ）---------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, head_dim); cos/sin: (1, 1, T, head_dim)
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    """RoPE 付き causal Multi-Head Attention（dropout なし）。"""

    def __init__(self, config: BabyloopConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

    def forward(self, x, cos, sin, attn_bias):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, is_causal=attn_bias is None
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: BabyloopConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=config.bias)
        self.down_proj = nn.Linear(config.ffn_hidden, config.d_model, bias=config.bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-Norm 残差ブロック: h += Attn(RMSNorm(h)); h += SwiGLU(RMSNorm(h))。"""

    def __init__(self, config: BabyloopConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_eps)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.rms_eps)
        self.mlp = SwiGLU(config)

    def forward(self, h, cos, sin, attn_bias):
        h = h + self.attn(self.attn_norm(h), cos, sin, attn_bias)
        h = h + self.mlp(self.mlp_norm(h))
        return h


# --- 視覚 connector（②④。frozen encoder の特徴を LM 空間へ射影する trainable projector）---
# ADR-0005/0006: encoder は data 側で事前計算、ここには torch 純正の projector のみ（自己完結維持）。


class VisualConnector(nn.Module):
    """事前計算済み視覚特徴 (B, V, vision_feature_dim) を d_model 空間へ射影する。

    - ``mlp``: 2層 MLP（vit_dim→d_model→d_model, GELU。LLaVA-1.5 流）。②の本命・trainable。
    - ``identity``: 射影なし（``vit_dim==d_model`` 前提）。射影なし＋未事前学習の劣化条件
      （ADR-0006 の naive）。学習パラメータを持たない。
    """

    def __init__(self, config: BabyloopConfig):
        super().__init__()
        self.connector_type = config.connector_type
        vit_dim = config.vision_feature_dim
        d = config.d_model
        if self.connector_type == "mlp":
            # projector は bias を持たせる（LM 本体の bias なし方針とは独立の trainable adapter）。
            self.fc1 = nn.Linear(vit_dim, d, bias=True)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(d, d, bias=True)
        elif self.connector_type == "identity":
            if vit_dim != d:
                raise ValueError(
                    f"connector_type=identity は vision_feature_dim==d_model が前提 "
                    f"({vit_dim} != {d})。ViT-B/14(768) を使うか mlp connector にする。"
                )
        else:
            raise ValueError(f"unknown connector_type: {self.connector_type}")

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        if self.connector_type == "identity":
            return vision_features
        return self.fc2(self.act(self.fc1(vision_features)))


# --- 出力コンテナ -----------------------------------------------------------


@dataclass
class LoopedModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    loop_hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class LoopedCausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    loop_hidden_states: tuple[torch.FloatTensor, ...] | None = None


# --- PreTrainedModel ラッパ -------------------------------------------------


class LoopedPreTrainedModel(PreTrainedModel):
    config_class = BabyloopConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)


class LoopedModel(LoopedPreTrainedModel):
    """重み共有 core ブロックを K 回反復するバックボーン（lm_head なし）。"""

    def __init__(self, config: BabyloopConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.prelude = nn.ModuleList(Block(config) for _ in range(config.n_prelude))
        self.core = nn.ModuleList(Block(config) for _ in range(config.n_core))
        self.coda = nn.ModuleList(Block(config) for _ in range(config.n_coda))
        self.final_norm = RMSNorm(config.d_model, config.rms_eps)

        # 視覚 connector（②④のみ）。fusion=None（①③ text）では構築しない＝パラメータ不変
        # （test_k1_equivalence / test_looped_adds_no_parameters を壊さない）。
        self.connector = VisualConnector(config) if config.fusion is not None else None
        if self.connector is not None and config.visual_inject_mode not in ("prefix_refresh", "broadcast"):
            raise ValueError(f"unknown visual_inject_mode: {config.visual_inject_mode}")

        head_dim = config.d_model // config.n_heads
        inv_freq = 1.0 / (
            config.rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _rope(self, T: int, device, dtype):
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]

    def _attn_bias(self, attention_mask, T, device, dtype):
        # padding が無ければ None を返し、SDPA の is_causal 経路（高速）に乗せる。
        if attention_mask is None or bool((attention_mask == 1).all()):
            return None
        causal = torch.ones(T, T, device=device, dtype=torch.bool).triu(1)
        key_pad = attention_mask.to(device) == 0  # (B, T)
        mask = causal[None, None] | key_pad[:, None, None, :]
        bias = torch.zeros(mask.shape, device=device, dtype=dtype)
        return bias.masked_fill(mask, torch.finfo(dtype).min)

    def inject(self, h, visual, loop_index: int):
        """④: ループ初期の各イテレーション先頭で視覚を再注入（``visual_inject_iters>0``）。

        Args:
            h: 現在の hidden states。``prefix_refresh`` では prefix 済 (B, V+T, d)、
               ``broadcast`` では (B, T, d)。
            visual: connector 射影済みの視覚特徴 (B, V, d)。n_visual は ``visual.size(1)`` から導出
                （pad 量の不一致バグを封じるため別経路で渡さない）。
            loop_index: 現在のループ回（0始まり、呼び出し側で ``< inject_iters`` を保証）。

        - ``prefix_refresh``（B 主系）: prefix 位置（先頭 V）へ視覚を再加算（右ゼロパディングして加算）。
          ②と同じ入れ方・同じ特徴で、変換で薄れた視覚信号を補充する（[ADR-0006]）。
        - ``broadcast``（A, ablation）: pooled 視覚ベクトルを全位置へブロードキャスト加算。系列長不変。
        """
        if self.config.visual_inject_mode == "broadcast":
            return h + visual.mean(dim=1, keepdim=True)
        # prefix_refresh: 先頭 V 位置だけに加算（残りは右ゼロパディングで no-op）。in-place 回避で autograd 安全。
        pad = h.size(1) - visual.size(1)  # = T（prefix_refresh は必ず prefix 済）
        return h + F.pad(visual, (0, 0, 0, pad))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        vision_features=None,
        output_hidden_states=False,
        **kwargs,
    ) -> LoopedModelOutput:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 視覚を connector で射影（ループ前に1回・prefix と in-loop で再利用）。
        # vision_features is None（テキスト・全 eval）なら全 skip＝①と完全同一経路（回帰不変条件）。
        # connector(nn.Linear) は autocast 下で bf16 を出すが embed_tokens(Embedding) は fp32 のまま。
        # cat/加算は dtype 昇格しないので embed の dtype に揃える（cuda bf16 で必須）。
        inject_iters = self.config.visual_inject_iters
        mode = self.config.visual_inject_mode
        visual = None
        if vision_features is not None and self.connector is not None:
            visual = self.connector(vision_features.to(inputs_embeds.dtype)).to(inputs_embeds.dtype)

        # prefix 連結: ②(inject_iters==0) と ④-B(prefix_refresh)。④-A(broadcast) は prefix しない。
        use_prefix = visual is not None and (inject_iters == 0 or mode == "prefix_refresh")
        n_visual = 0
        if use_prefix:
            n_visual = visual.size(1)
            inputs_embeds = torch.cat([visual, inputs_embeds], dim=1)
            if attention_mask is not None:
                visual_mask = attention_mask.new_ones((attention_mask.size(0), n_visual))
                attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        h = inputs_embeds
        # inject_input（③のループ機構＝毎ループ入力を再加算）は**テキストのみ**再加算する。
        # 視覚 prefix の再注入は inject() が単独で担い、両者の役割を分離する（ADR-0006）。
        # → prefix 済なら residual の視覚位置を 0 に（視覚 prefix が inject_input で二重再加算されるのを防ぐ）。
        residual_input = inputs_embeds  # ③（視覚なし）はそのまま＝従来挙動不変
        if n_visual > 0:
            residual_input = F.pad(inputs_embeds[:, n_visual:], (0, 0, n_visual, 0))
        B, T, _ = h.shape
        cos, sin = self._rope(T, h.device, h.dtype)
        attn_bias = self._attn_bias(attention_mask, T, h.device, h.dtype)

        # ④: in-loop 注入を行うか（inject_iters>0 かつ視覚あり）。②(inject_iters==0)は False。
        do_inject = visual is not None and inject_iters > 0

        all_hidden = [h] if output_hidden_states else None
        loop_hidden = []

        for block in self.prelude:
            h = block(h, cos, sin, attn_bias)
            if output_hidden_states:
                all_hidden.append(h)
        for loop_index in range(self.config.k):
            # 各イテレーションの先頭で再注入（リフレッシュした視覚をこのイテレーションの計算に乗せる）。
            if do_inject and loop_index < inject_iters:
                h = self.inject(h, visual, loop_index)
            for block in self.core:
                h = block(h, cos, sin, attn_bias)
                if output_hidden_states:
                    all_hidden.append(h)
            if self.config.inject_input:
                h = h + residual_input
            loop_hidden.append(h)
        for block in self.coda:
            h = block(h, cos, sin, attn_bias)
            if output_hidden_states:
                all_hidden.append(h)

        h = self.final_norm(h)
        return LoopedModelOutput(
            last_hidden_state=h,
            hidden_states=tuple(all_hidden) if output_hidden_states else None,
            loop_hidden_states=tuple(loop_hidden),
        )


class LoopedForCausalLM(LoopedPreTrainedModel):
    """言語モデリングヘッド付き（``AutoModelForCausalLM`` 互換）。"""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BabyloopConfig):
        super().__init__(config)
        self.model = LoopedModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new):
        self.lm_head = new

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        vision_features=None,
        labels=None,
        output_hidden_states=False,
        **kwargs,
    ) -> LoopedCausalLMOutput:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            vision_features=vision_features,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(out.last_hidden_state)

        loss = None
        if labels is not None:
            # ②: 視覚 prefix が付くと logits 系列長 (V+L) が labels 長 (L) より長い。
            # 差分 V 個の視覚位置を -100 で前置し、視覚位置では LM loss を取らない（テキストのみ）。
            n_prefix = logits.size(1) - labels.size(1)
            if n_prefix > 0:
                pad = labels.new_full((labels.size(0), n_prefix), -100)
                labels = torch.cat([pad, labels], dim=1)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return LoopedCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=out.hidden_states,
            loop_hidden_states=out.loop_hidden_states,
        )


# 公式 evaluation-pipeline が trust_remote_code で AutoModel 系から読めるよう登録。
# save_pretrained 時に auto_map と本ファイル群が checkpoint へ複製される。
BabyloopConfig.register_for_auto_class()
LoopedModel.register_for_auto_class("AutoModel")
LoopedForCausalLM.register_for_auto_class("AutoModelForCausalLM")
