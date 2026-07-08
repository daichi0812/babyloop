"""② 視覚 prefix の forward 不変条件（ADR-0006 / docs/design.md §4）。

ロックする不変条件:
  1. **視覚なし forward は構造的 no-op**: connector を持つ MM モデルでも
     ``vision_features=None`` ならテキスト計算を一切変えない（embed→core→head の参照と一致）。
     ②④の ckpt を BLiMP（テキストのみ）で読んでも落ちない前提（最重要・回帰不変）。
     ※「②重み==①重み」の主張ではない（connector init が RNG を消費し重みは別物）。
  2. prefix で系列長が V+L になり、loss は **テキスト位置のみ**（視覚位置は -100）。
     モデルの自動 -100 前置が、手動で -100×V を前置した場合と一致する。
  3. fusion=None（①③）は connector を持たない＝視覚フィールドを足してもパラメータ不変。
"""

import torch

from babyloop.models.configuration_babyloop import BabyloopConfig
from babyloop.models.modeling_babyloop import LoopedForCausalLM

D, NC, NH, FF, VOCAB, MAXLEN, FEAT = 64, 2, 4, 128, 64, 64, 32


def _mm_model(connector_type="mlp", n_inject=0, k=1):
    torch.manual_seed(0)
    cfg = BabyloopConfig(
        d_model=D, n_layers=NC, n_heads=NH, ffn_hidden=FF, n_prelude=0, n_core=NC,
        n_coda=0, k=k, vocab_size=VOCAB, max_seq_len=MAXLEN,
        fusion="connector", vision_feature_dim=FEAT, connector_type=connector_type,
        n_visual_tokens=4, visual_inject_iters=n_inject,
    )
    return LoopedForCausalLM(cfg).eval()


def _text_reference(model, input_ids):
    """connector を介さず embed→prelude→core×k→coda→final_norm→lm_head を直接適用する参照。"""
    mm = model.model
    h = mm.embed_tokens(input_ids)
    cos, sin = mm._rope(h.size(1), h.device, h.dtype)
    bias = mm._attn_bias(None, h.size(1), h.device, h.dtype)
    for block in mm.prelude:
        h = block(h, cos, sin, bias)
    for _ in range(mm.config.k):
        for block in mm.core:
            h = block(h, cos, sin, bias)
    for block in mm.coda:
        h = block(h, cos, sin, bias)
    h = mm.final_norm(h)
    return model.lm_head(h)


def test_vision_none_is_structural_noop():
    """connector を持つ MM モデルでも vision_features=None はテキスト計算を変えない。"""
    model = _mm_model()
    assert model.model.connector is not None, "MM モデルは connector を持つ"
    ids = torch.randint(0, VOCAB, (2, 10))
    with torch.no_grad():
        none_logits = model(input_ids=ids, vision_features=None).logits
        ref = _text_reference(model, ids)
    assert none_logits.shape == (2, 10, VOCAB)
    assert torch.allclose(none_logits, ref, atol=1e-6)


def test_vision_prefix_changes_output_and_grows_sequence():
    """vision_features を与えると connector 経由で prefix され、系列長が V+L になる。"""
    model = _mm_model()
    ids = torch.randint(0, VOCAB, (2, 10))
    vf = torch.randn(2, 5, FEAT)  # V=5 視覚トークン
    with torch.no_grad():
        none_logits = model(input_ids=ids, vision_features=None).logits
        pre_logits = model(input_ids=ids, vision_features=vf).logits
    assert pre_logits.shape == (2, 5 + 10, VOCAB), "系列長 = V + L"
    # テキスト位置の logits は視覚 prefix の文脈を受けて None 経路と変わる（融合が効いている）。
    assert not torch.allclose(pre_logits[:, 5:, :], none_logits, atol=1e-4)


def test_prefix_label_mask_is_text_only():
    """loss はテキスト位置のみ。モデルの自動 -100 前置 == 手動 -100×V 前置。"""
    model = _mm_model()
    B, L, V = 2, 8, 5
    caption = torch.randint(0, VOCAB, (B, L))
    vf = torch.randn(B, V, FEAT)
    loss_auto = model(input_ids=caption, vision_features=vf, labels=caption).loss
    manual = torch.cat([caption.new_full((B, V), -100), caption], dim=1)
    loss_manual = model(input_ids=caption, vision_features=vf, labels=manual).loss
    assert torch.isfinite(loss_auto)
    assert torch.allclose(loss_auto, loss_manual, atol=1e-6)


def test_attention_mask_extended_for_prefix():
    """attention_mask を与えても prefix 分拡張されて落ちない（padding ありの caption バッチ）。"""
    model = _mm_model()
    B, L, V = 2, 8, 5
    caption = torch.randint(0, VOCAB, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)
    attn[0, -3:] = 0  # padding
    vf = torch.randn(B, V, FEAT)
    out = model(input_ids=caption, attention_mask=attn, vision_features=vf, labels=caption)
    assert out.logits.shape == (B, V + L, VOCAB)
    assert torch.isfinite(out.loss)


def test_autocast_visual_prefix_dtype_match():
    """autocast(bf16) 下で connector(Linear)出力と embed(fp32)の cat が dtype 不一致で落ちないこと。

    cuda の AMP 経路（trainer は cuda 時のみ autocast）を CPU bf16 で代理検証する。
    回帰: visual_embeds を embed の dtype へ揃えないと torch.cat が RuntimeError（②の主経路 mlp）。
    """
    model = _mm_model(connector_type="mlp")
    ids = torch.randint(0, VOCAB, (2, 6))
    vf = torch.randn(2, 4, FEAT)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(input_ids=ids, vision_features=vf, labels=ids)
    assert torch.isfinite(out.loss)


def test_mm_checkpoint_roundtrip_trust_remote_code(tmp_path):
    """connector 付きモデルが save_pretrained → AutoModelForCausalLM(trust_remote_code) で復元でき、
    テキスト(vision_features=None)経路の logits が保存前と一致すること。

    ②の完成条件「ckpt が公式 pipeline を通って BLiMP が出る」の最も近いローカル代理。
    新 config フィールド・connector 再構築・state_dict キーが trust_remote_code ロードを壊さないことを pin。
    """
    from transformers import AutoModelForCausalLM

    model = _mm_model(connector_type="mlp")
    ids = torch.randint(0, VOCAB, (2, 7))
    with torch.no_grad():
        before = model(input_ids=ids, vision_features=None).logits
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True).eval()
    assert loaded.config.fusion is not None and loaded.model.connector is not None
    with torch.no_grad():
        after = loaded(input_ids=ids, vision_features=None).logits
    assert torch.allclose(before, after, atol=1e-5)


def test_text_model_has_no_connector_params():
    """fusion=None（①③）は connector を持たず、視覚フィールドを足してもパラメータ不変。"""
    torch.manual_seed(0)
    base = dict(d_model=D, n_layers=NC, n_heads=NH, ffn_hidden=FF, n_prelude=0,
                n_core=NC, n_coda=0, k=1, vocab_size=VOCAB, max_seq_len=MAXLEN)
    text = LoopedForCausalLM(BabyloopConfig(**base))
    assert text.model.connector is None
    assert not any("connector" in k for k in text.state_dict())

    mm = _mm_model()
    assert any("connector" in k for k in mm.state_dict())


def test_identity_connector_requires_dim_match():
    """connector_type=identity は vision_feature_dim==d_model 前提（不一致なら明確に失敗）。"""
    import pytest
    with pytest.raises(ValueError):
        LoopedForCausalLM(BabyloopConfig(
            d_model=D, n_layers=NC, n_heads=NH, ffn_hidden=FF, n_prelude=0, n_core=NC,
            n_coda=0, k=1, vocab_size=VOCAB, max_seq_len=MAXLEN,
            fusion="connector", vision_feature_dim=FEAT, connector_type="identity",
        ))
    # 一致すれば identity は構築でき、射影なし（パラメータ無し）。
    m = LoopedForCausalLM(BabyloopConfig(
        d_model=D, n_layers=NC, n_heads=NH, ffn_hidden=FF, n_prelude=0, n_core=NC,
        n_coda=0, k=1, vocab_size=VOCAB, max_seq_len=MAXLEN,
        fusion="connector", vision_feature_dim=D, connector_type="identity",
    ))
    assert not any("connector" in k for k in m.state_dict()), "identity は学習パラメータを持たない"
