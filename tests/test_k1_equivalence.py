"""K=1 の LoopedTransformer ≡ 標準Transformer の等価性テスト。

2×2デザインの要因A（標準 vs 再帰）を ``k`` だけに帰属させられること（ADR-0001）が
本テストの責務。①(standard, k=1, inject_input=false) と ③(looped, k=4,
inject_input=true) の差は **ループ回数と入力注入だけで、パラメータは一切増えない**
ことを固定する。これが崩れると「①と③の差はループだけ」という主張が成り立たない。

ロックする不変条件:
  1. k=1 のループ forward は、core を1回だけ通す標準Transformerと数値一致（logits/grad）。
  2. ループ回数 k が core ブロックの適用回数を厳密に駆動する（k=1→各1回, k=4→各4回）。
     → k=1 が「標準の深さ」、k>1 が「同一重みの反復」であることの非循環な構造的保証。
  3. k や inject_input を変えてもパラメータ集合（キー・形状・値）は不変＝重み共有で増えない。
"""

import torch

from babyloop.models.configuration_babyloop import BabyloopConfig
from babyloop.models.modeling_babyloop import LoopedForCausalLM

N_CORE = 3


def _build(k: int, inject_input: bool = False) -> LoopedForCausalLM:
    # 同一seedなので k/inject_input に依らず重み初期化は同一（モジュール構成が同じ）。
    torch.manual_seed(0)
    cfg = BabyloopConfig(
        d_model=64, n_layers=N_CORE, n_heads=4, ffn_hidden=128,
        n_prelude=0, n_core=N_CORE, n_coda=0, k=k, inject_input=inject_input,
        vocab_size=128, max_seq_len=32,
    )
    return LoopedForCausalLM(cfg).eval()


def _reference_no_loop(model: LoopedForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """同一サブモジュールを使い、ループ構文を介さず core を1回だけ適用する参照。"""
    mm = model.model
    h = mm.embed_tokens(input_ids)
    cos, sin = mm._rope(h.size(1), h.device, h.dtype)
    bias = mm._attn_bias(None, h.size(1), h.device, h.dtype)
    for block in mm.core:
        h = block(h, cos, sin, bias)
    h = mm.final_norm(h)
    return model.lm_head(h)


def test_k1_logits_match_standard():
    """同一重み・同一入力で K=1 のlogitsが標準（無ループ）実装と allclose になること。"""
    model = _build(k=1)
    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        looped = model(input_ids=ids).logits
        reference = _reference_no_loop(model, ids)
    assert torch.allclose(looped, reference, atol=1e-6)

    # ループが実際に効いていること（テストが空虚でないことの保証）: K=2 は異なる。
    with torch.no_grad():
        model.model.config.k = 2
        looped_k2 = model(input_ids=ids).logits
    assert not torch.allclose(looped_k2, reference, atol=1e-4)


def test_k1_gradients_match_standard():
    """K=1 で逆伝播の勾配も標準（無ループ）実装と一致すること。"""
    ids = torch.randint(0, 128, (2, 16))

    model = _build(k=1)
    model(input_ids=ids).logits.pow(2).sum().backward()
    g_looped = model.get_input_embeddings().weight.grad.clone()

    model.zero_grad(set_to_none=True)
    _reference_no_loop(model, ids).pow(2).sum().backward()
    g_reference = model.get_input_embeddings().weight.grad.clone()

    assert torch.allclose(g_looped, g_reference, atol=1e-5)


def _count_core_invocations(model: LoopedForCausalLM, ids: torch.Tensor) -> list[int]:
    counts = [0] * len(model.model.core)
    handles = [
        block.register_forward_hook(lambda *_a, _i=i: counts.__setitem__(_i, counts[_i] + 1))
        for i, block in enumerate(model.model.core)
    ]
    with torch.no_grad():
        model(input_ids=ids)
    for h in handles:
        h.remove()
    return counts


def test_loop_count_drives_core_invocations():
    """ループ回数 k が core 適用回数を厳密に駆動すること（k=1→各1回, k=4→各4回）。

    参照 forward とのモジュール共有による循環を補う非循環な構造テスト。
    k=1 が標準の深さ（各 core 1回）、k>1 が同一重みの反復であることを直接示す。
    """
    ids = torch.randint(0, 128, (2, 16))
    for k in (1, 2, 4):
        counts = _count_core_invocations(_build(k=k), ids)
        assert counts == [k] * N_CORE, f"k={k}: expected each core x{k}, got {counts}"


def test_looped_adds_no_parameters():
    """①(k=1,inject=false) と ③(k=4,inject=true) でパラメータが完全一致すること。

    重み共有により k を増やしてもパラメータは増えない＝「効果的な深さを
    データ・パラメータを増やさずに足す」という研究主張の前提。
    """
    std = _build(k=1, inject_input=False)   # ①
    loop = _build(k=4, inject_input=True)   # ③

    sd_std, sd_loop = std.state_dict(), loop.state_dict()
    assert sd_std.keys() == sd_loop.keys(), "state_dict のキー集合が一致すること"
    for key in sd_std:
        assert sd_std[key].shape == sd_loop[key].shape, f"{key}: 形状一致"
        assert torch.equal(sd_std[key], sd_loop[key]), f"{key}: 値一致（kは重みに影響しない）"

    n_std = sum(p.numel() for p in std.parameters())
    n_loop = sum(p.numel() for p in loop.parameters())
    assert n_std == n_loop, f"パラメータ数が一致すること: {n_std} != {n_loop}"
