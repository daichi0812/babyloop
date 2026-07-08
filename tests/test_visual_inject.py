"""④ 視覚 in-loop 注入 inject() の不変条件（ADR-0006 / docs/design.md §4）。

主系 B（prefix_refresh: prefix + ループ初期 N 回の各先頭で prefix 位置へ再加算）と
フラグ A（broadcast: pooled 視覚を全位置へ加算）。ロックする不変条件:
  1. forward 再構成で①(vision=None)・②(inject_iters=0, prefix)経路を踏み外していない（**数値一致**）。
  2. B/A とも forward/backward が通り、注入が実際に h を変える（no-op バグ検出）。
  3. 数 optimizer step で loss が発散しない（N 回加算×RMSNorm のスケール安定性）。
  4. autocast(bf16) で dtype 不一致で落ちない（cuda AMP 代理）。
"""

import torch
import torch.nn.functional as F

from babyloop.models.configuration_babyloop import BabyloopConfig
from babyloop.models.modeling_babyloop import LoopedForCausalLM

D, NC, NH, FF, VOCAB, MAXLEN, FEAT, V = 64, 2, 4, 128, 64, 64, 32, 4


def _model(k=2, inject_iters=0, mode="prefix_refresh"):
    torch.manual_seed(0)  # seed 固定 → inject_iters/mode に依らず重みは同一
    cfg = BabyloopConfig(
        d_model=D, n_layers=NC, n_heads=NH, ffn_hidden=FF, n_prelude=0, n_core=NC, n_coda=0,
        k=k, vocab_size=VOCAB, max_seq_len=MAXLEN, fusion="connector", vision_feature_dim=FEAT,
        connector_type="mlp", n_visual_tokens=V, visual_inject_iters=inject_iters,
        visual_inject_mode=mode,
    )
    return LoopedForCausalLM(cfg).eval()


def _ref(model, ids, k, vf=None):
    """connector を介す prefix（任意）＋ core×k（注入なし）の参照。n_prelude=n_coda=0, inject_input=False。"""
    mm = model.model
    h = mm.embed_tokens(ids)
    if vf is not None:
        visual = mm.connector(vf.to(h.dtype)).to(h.dtype)
        h = torch.cat([visual, h], dim=1)
    cos, sin = mm._rope(h.size(1), h.device, h.dtype)
    bias = mm._attn_bias(None, h.size(1), h.device, h.dtype)
    for _ in range(k):
        for block in mm.core:
            h = block(h, cos, sin, bias)
    h = mm.final_norm(h)
    return model.lm_head(h)


def test_vision_none_equals_text_reference_kgt1():
    """①等価（数値一致）: vision_features=None & K>1 が無注入のテキスト参照と allclose。"""
    model = _model(k=3, inject_iters=2)  # inject_iters>0 でも vision=None なら注入経路に入らない
    ids = torch.randint(0, VOCAB, (2, 10))
    with torch.no_grad():
        none_logits = model(input_ids=ids, vision_features=None).logits
        ref = _ref(model, ids, k=3, vf=None)
    assert torch.allclose(none_logits, ref, atol=1e-6)


def test_inject_iters0_equals_prefix_loop_reference():
    """②回帰（数値一致）: inject_iters=0 + 視覚 prefix が prefix→core×k（注入なし）参照と allclose。"""
    model = _model(k=3, inject_iters=0)
    ids = torch.randint(0, VOCAB, (2, 8))
    vf = torch.randn(2, V, FEAT)
    with torch.no_grad():
        logits = model(input_ids=ids, vision_features=vf).logits
        ref = _ref(model, ids, k=3, vf=vf)
    assert logits.shape == (2, V + 8, VOCAB)
    assert torch.allclose(logits, ref, atol=1e-6)


def test_prefix_refresh_forward_backward_and_changes_output():
    """B: K>1・inject_iters>0 で forward/backward、系列 V+T、connector に勾配、注入が h を変える。"""
    L = 8
    ids = torch.randint(0, VOCAB, (2, L))
    vf = torch.randn(2, V, FEAT)

    m2 = _model(k=2, inject_iters=2)
    out = m2(input_ids=ids, vision_features=vf, labels=ids)
    assert out.logits.shape == (2, V + L, VOCAB)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for n, p in m2.named_parameters() if "connector" in n]
    assert grads and all(g is not None for g in grads)

    # 同一重み（seed固定）で inject_iters=0 と比べ、再注入が出力を変える（no-op バグ検出）。
    m0 = _model(k=2, inject_iters=0)
    with torch.no_grad():
        assert not torch.allclose(
            m2(input_ids=ids, vision_features=vf).logits,
            m0(input_ids=ids, vision_features=vf).logits,
            atol=1e-4,
        )


def test_broadcast_mode_no_prefix_and_effective():
    """A: broadcast は prefix せず系列 T、loss 有限、注入が効く。"""
    L = 8
    ids = torch.randint(0, VOCAB, (2, L))
    vf = torch.randn(2, V, FEAT)
    m = _model(k=2, inject_iters=2, mode="broadcast")
    out = m(input_ids=ids, vision_features=vf, labels=ids)
    assert out.logits.shape == (2, L, VOCAB), "broadcast は prefix しない＝系列長 T"
    assert torch.isfinite(out.loss)
    # inject_iters=0(broadcast でも prefix しない)と比べ注入が効く。
    m0 = _model(k=2, inject_iters=0, mode="broadcast")
    with torch.no_grad():
        # inject_iters=0 は broadcast でも prefix（use_prefix= inject_iters==0）→系列 V+L になる点に注意。
        # ここでは「broadcast 注入が効く」を inject_iters 1 vs 2 の差で見る。
        a1 = _model(k=2, inject_iters=1, mode="broadcast")(input_ids=ids, vision_features=vf).logits
        a2 = m(input_ids=ids, vision_features=vf).logits
    assert not torch.allclose(a1, a2, atol=1e-4)


def _run_steps(model, n_steps=5, prefix=True):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for s in range(n_steps):
        torch.manual_seed(100 + s)
        ids = torch.randint(0, VOCAB, (2, 8))
        vf = torch.randn(2, V, FEAT)
        loss = model(input_ids=ids, vision_features=vf, labels=ids).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_stability_few_steps_prefix_refresh():
    """B: 数 optimizer step で loss が有限・非発散（N 回加算のスケール暴れを捕捉）。"""
    losses = _run_steps(_model(k=2, inject_iters=2, mode="prefix_refresh"))
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert losses[-1] < losses[0] * 5, f"loss が発散気味: {losses}"


def test_stability_few_steps_broadcast():
    """A: 同上（broadcast）。"""
    losses = _run_steps(_model(k=2, inject_iters=2, mode="broadcast"))
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert losses[-1] < losses[0] * 5, f"loss が発散気味: {losses}"


def test_loop_mm_real_config_inject_input_true():
    """loop_mm の実構成（inject_input=true + 視覚 prefix + inject_iters>0）で forward/backward が通る。

    テストヘルパの既定 inject_input=False では踏まない経路。inject_input=true だと residual_input は
    prefix 済 [visual|text] で、視覚 prefix も毎ループ再加算される（その挙動をここで pin）。
    """
    torch.manual_seed(0)
    cfg = BabyloopConfig(
        d_model=D, n_layers=NC, n_heads=NH, ffn_hidden=FF, n_prelude=0, n_core=NC, n_coda=0,
        k=4, inject_input=True, vocab_size=VOCAB, max_seq_len=MAXLEN, fusion="connector",
        vision_feature_dim=FEAT, connector_type="mlp", n_visual_tokens=V,
        visual_inject_iters=2, visual_inject_mode="prefix_refresh",
    )
    model = LoopedForCausalLM(cfg)
    ids = torch.randint(0, VOCAB, (2, 8))
    vf = torch.randn(2, V, FEAT)
    out = model(input_ids=ids, vision_features=vf, labels=ids)
    assert out.logits.shape == (2, V + 8, VOCAB)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if "connector" in n)


def test_inject_input_re_adds_text_only_not_visual_prefix():
    """inject_input は**テキストのみ**再加算（視覚 prefix 位置は 0）。loop_mm_prefix が真の『視覚once』。

    inject_input=true + prefix + inject_iters=0 の forward が「core×k 後に [0|text] を毎ループ再加算」
    の参照と allclose。residual に視覚が混ざっていれば（旧挙動）一致しない＝視覚 prefix の二重再加算を pin。
    """
    torch.manual_seed(0)
    cfg = BabyloopConfig(
        d_model=D, n_layers=NC, n_heads=NH, ffn_hidden=FF, n_prelude=0, n_core=NC, n_coda=0,
        k=3, inject_input=True, vocab_size=VOCAB, max_seq_len=MAXLEN, fusion="connector",
        vision_feature_dim=FEAT, connector_type="mlp", n_visual_tokens=V, visual_inject_iters=0,
    )
    model = LoopedForCausalLM(cfg).eval()
    ids = torch.randint(0, VOCAB, (2, 8))
    vf = torch.randn(2, V, FEAT)

    mm = model.model
    text = mm.embed_tokens(ids)
    visual = mm.connector(vf.to(text.dtype)).to(text.dtype)
    h = torch.cat([visual, text], dim=1)
    residual_text = F.pad(text, (0, 0, V, 0))  # [0×V | text]
    cos, sin = mm._rope(h.size(1), h.device, h.dtype)
    bias = mm._attn_bias(None, h.size(1), h.device, h.dtype)
    for _ in range(3):
        for block in mm.core:
            h = block(h, cos, sin, bias)
        h = h + residual_text  # inject_input = テキストのみ
    ref = model.lm_head(mm.final_norm(h))

    with torch.no_grad():
        logits = model(input_ids=ids, vision_features=vf).logits
    assert torch.allclose(logits, ref, atol=1e-6)


def test_unknown_inject_mode_rejected():
    """未知の visual_inject_mode は構築時に弾く（silent fallthrough しない）。"""
    import pytest
    with pytest.raises(ValueError):
        LoopedForCausalLM(BabyloopConfig(
            d_model=D, n_layers=NC, n_heads=NH, ffn_hidden=FF, n_prelude=0, n_core=NC, n_coda=0,
            k=2, vocab_size=VOCAB, max_seq_len=MAXLEN, fusion="connector",
            vision_feature_dim=FEAT, connector_type="mlp", visual_inject_mode="bogus",
        ))


def test_autocast_inject_dtype_match():
    """autocast(bf16) で B/A とも cat/加算の dtype 不一致で落ちないこと（cuda AMP 代理）。"""
    ids = torch.randint(0, VOCAB, (2, 6))
    vf = torch.randn(2, V, FEAT)
    for mode in ("prefix_refresh", "broadcast"):
        m = _model(k=2, inject_iters=2, mode=mode)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = m(input_ids=ids, vision_features=vf, labels=ids)
        assert torch.isfinite(out.loss), mode
