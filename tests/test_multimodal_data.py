"""② multimodal データ経路（preprocessing / dataloader）の検証。

- MultimodalPreprocessor: caption 語数の budget 計上・特徴なし caption のスキップ・text/caption 分割。
- MultimodalDataModule: homogeneous バッチの collate 形（caption は vision_features・pad→-100、
  text は input_ids==labels）、保存トークン→n_visual_tokens の再プール、ratio によるストリーム選択。
- caption バッチ → モデル forward/backward が通り connector に勾配が流れる（end-to-end 配線）。
"""

import json

import numpy as np
import torch
from omegaconf import OmegaConf

from babyloop.data.dataloader import MultimodalDataModule, _pool_visual
from babyloop.data.preprocessing import MultimodalPreprocessor, mm_caption_dir, mm_text_dir
from babyloop.models.configuration_babyloop import BabyloopConfig
from babyloop.models.modeling_babyloop import LoopedForCausalLM

SEQ, STORED, DIM = 8, 4, 8


def _write_processed(root, n_text_blocks=6, n_caps=5, n_images=5):
    """合成済み MM 前処理成果物（text packed / caption records / vision feats）を書く。"""
    proc = root / "processed"
    feat = root / "vision"
    (proc / "text").mkdir(parents=True)
    (proc / "captions").mkdir(parents=True)
    feat.mkdir(parents=True)

    # text packed blocks
    tokens = np.arange(n_text_blocks * SEQ, dtype=np.uint16) % 50
    np.save(proc / "text" / "tokens.npy", tokens)
    np.save(proc / "text" / "block_words.npy", np.full(n_text_blocks, 7, dtype=np.int64))
    (proc / "text" / "meta.json").write_text(json.dumps(
        {"max_seq_len": SEQ, "n_blocks": n_text_blocks, "words_per_token": 0.9}))

    # caption records（可変長）
    lens = [3, 4, 2, 5, 3][:n_caps]
    flat, offsets = [], [0]
    for L in lens:
        flat.append(np.arange(L, dtype=np.uint16) % 50)
        offsets.append(offsets[-1] + L)
    np.save(proc / "captions" / "caption_tokens.npy", np.concatenate(flat))
    np.save(proc / "captions" / "caption_offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(proc / "captions" / "caption_vision_rows.npy",
            np.arange(n_caps, dtype=np.int64) % n_images)
    np.save(proc / "captions" / "caption_words.npy", np.asarray(lens, dtype=np.int64))

    # vision feats memmap
    feats = np.random.RandomState(0).randn(n_images, STORED, DIM).astype(np.float16)
    np.save(feat / "feats.npy", feats)
    (feat / "index.json").write_text(json.dumps(
        {"rows": {f"img{i}": i for i in range(n_images)}}))
    return proc, feat


def _cfgs(proc, feat, ratio=0.5, batch_size=2, seed=0):
    data_cfg = OmegaConf.create({
        "processed_dir": str(proc), "feature_dir": str(feat), "text_caption_ratio": ratio,
    })
    train_cfg = OmegaConf.create({"batch_size": batch_size, "seed": seed})
    return data_cfg, train_cfg


def test_pool_visual_math():
    x = torch.randn(2, 16, DIM)
    assert _pool_visual(x, 4).shape == (2, 4, DIM)        # 4×4 → 2×2
    assert _pool_visual(x, 16).shape == (2, 16, DIM)      # 同数は no-op
    assert torch.equal(_pool_visual(x, 0), x)             # 0 は no-op


def test_caption_collate_shapes_and_label_mask(tmp_path):
    proc, feat = _write_processed(tmp_path)
    data_cfg, train_cfg = _cfgs(proc, feat, ratio=0.0)  # caption のみ
    dm = MultimodalDataModule(data_cfg, train_cfg, n_visual_tokens=STORED, pad_token_id=0)
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert set(batch) >= {"input_ids", "attention_mask", "labels", "vision_features", "batch_words"}
    B, L = batch["input_ids"].shape
    assert batch["vision_features"].shape == (B, STORED, DIM)
    # pad 位置（attention_mask==0）は labels が -100。
    pad = batch["attention_mask"] == 0
    assert (batch["labels"][pad] == -100).all()
    assert (batch["labels"][~pad] == batch["input_ids"][~pad]).all()


def test_text_collate_is_lm(tmp_path):
    proc, feat = _write_processed(tmp_path)
    data_cfg, train_cfg = _cfgs(proc, feat, ratio=1.0)  # text のみ
    dm = MultimodalDataModule(data_cfg, train_cfg, n_visual_tokens=STORED, pad_token_id=0)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert "vision_features" not in batch
    assert torch.equal(batch["input_ids"], batch["labels"])
    assert batch["input_ids"].shape[1] == SEQ


def test_mixed_loader_respects_ratio(tmp_path):
    proc, feat = _write_processed(tmp_path)
    # ratio=1.0 → text のみ（vision なし）。ratio=0.0 → caption のみ（vision あり）。
    for ratio, expect_vision in [(1.0, False), (0.0, True)]:
        data_cfg, train_cfg = _cfgs(proc, feat, ratio=ratio)
        dm = MultimodalDataModule(data_cfg, train_cfg, n_visual_tokens=STORED, pad_token_id=0)
        dm.setup()
        it = iter(dm.train_dataloader())
        for _ in range(5):
            assert ("vision_features" in next(it)) == expect_vision


def test_visual_pooling_in_collate(tmp_path):
    """stored_tokens(4) → n_visual_tokens(1) の再プールが collate で効く。"""
    proc, feat = _write_processed(tmp_path)
    data_cfg, train_cfg = _cfgs(proc, feat, ratio=0.0)
    dm = MultimodalDataModule(data_cfg, train_cfg, n_visual_tokens=1, pad_token_id=0)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch["vision_features"].shape[1] == 1


def test_caption_batch_end_to_end(tmp_path):
    """caption バッチ → モデル forward/backward が通り connector に勾配が流れる。"""
    proc, feat = _write_processed(tmp_path)
    data_cfg, train_cfg = _cfgs(proc, feat, ratio=0.0)
    dm = MultimodalDataModule(data_cfg, train_cfg, n_visual_tokens=STORED, pad_token_id=0)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))

    torch.manual_seed(0)
    model = LoopedForCausalLM(BabyloopConfig(
        d_model=16, n_layers=2, n_heads=2, ffn_hidden=32, n_prelude=0, n_core=2, n_coda=0,
        k=1, vocab_size=50, max_seq_len=64, fusion="connector", vision_feature_dim=DIM,
        connector_type="mlp", n_visual_tokens=STORED,
    ))
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                vision_features=batch["vision_features"], labels=batch["labels"])
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "connector" in n]
    assert grads and all(g is not None for g in grads), "connector に勾配が流れること"


# --- MultimodalPreprocessor（特徴なし caption のスキップ・budget 計上）---


def test_precompute_official_v1_shape_and_index(tmp_path):
    """precompute_vision --feature-source official: OSF の (N,768) .npy → (N,1,768) memmap + index。

    V=1 reshape・CC 2分割(.npy)の連番連結・源_行index の index 対応を pin。
    """
    import importlib.util
    from pathlib import Path as P
    from types import SimpleNamespace

    osf = tmp_path / "osf"
    osf.mkdir()
    rng = np.random.RandomState(0)
    np.save(osf / "local_narr_dino_v2_states.npy", rng.randn(3, 768).astype(np.float32))
    np.save(osf / "cc_3M_dino_v2_states_1of2.npy", rng.randn(2, 768).astype(np.float32))
    np.save(osf / "cc_3M_dino_v2_states_2of2.npy", rng.randn(2, 768).astype(np.float32))

    root = P(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("precompute_vision", root / "scripts" / "precompute_vision.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    feat = tmp_path / "vision"
    mod._from_official(SimpleNamespace(official_dir=str(osf), feature_dir=str(feat),
                                       sources="ln,cc", max_rows=None))

    feats = np.load(feat / "feats.npy")
    assert feats.shape == (7, 1, 768), "V=1（N,1,768）"
    assert feats.dtype == np.float16
    rows = json.loads((feat / "index.json").read_text())["rows"]
    # LN 先（ln_0..2）→ CC（cc_0..3、2分割を連番連結）。
    assert rows["ln_0"] == 0 and rows["ln_2"] == 2
    assert rows["cc_0"] == 3 and rows["cc_3"] == 6


def test_preprocessor_tail_truncates_long_captions(tmp_path):
    """長文 caption は token 列を tail-truncate（prefix V + caption ≤ max_seq_len）。"""
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "train.txt").write_text(
        "\n".join("the quick brown fox jumps over the lazy dog" for _ in range(40))
    )
    cap_dir = tmp_path / "captions"
    cap_dir.mkdir()
    long_cap = " ".join(["word"] * 80)  # 80語 → max_seq_len(16) を確実に超える
    (cap_dir / "captions.jsonl").write_text(json.dumps({"image_id": "img0", "caption": long_cap}))
    feat_dir = tmp_path / "vision"
    feat_dir.mkdir()
    (feat_dir / "index.json").write_text(json.dumps({"rows": {"img0": 0}}))

    cfg = OmegaConf.create({
        "model": {"vocab_size": 300, "max_seq_len": 16, "vision_encoder": {"n_visual_tokens": 1}},
        "data": {
            "modality": "multimodal", "word_budget": 4000, "text_caption_ratio": 0.5,
            "text_dir": str(text_dir), "caption_dir": str(cap_dir), "feature_dir": str(feat_dir),
            "processed_dir": str(tmp_path / "processed"), "tokenizer_path": str(tmp_path / "tok"),
        },
    })
    MultimodalPreprocessor(cfg).run()

    cap_meta = json.loads((mm_caption_dir(cfg.data) / "meta.json").read_text())
    assert cap_meta["cap_max_len"] == 15, "max_seq_len(16) - n_visual(1)"
    assert cap_meta["n_truncated"] == 1
    offsets = np.load(mm_caption_dir(cfg.data) / "caption_offsets.npy")
    assert int(offsets[1] - offsets[0]) == 15, "格納 caption 長 = cap_max_len に truncate"


def test_preprocessor_skips_captions_without_features(tmp_path):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    # BPE 学習＋ text packing 用に十分な行。
    (text_dir / "train.txt").write_text(
        "\n".join("the quick brown fox jumps over the lazy dog near the river" for _ in range(40))
    )
    cap_dir = tmp_path / "captions"
    cap_dir.mkdir()
    caps = [{"image_id": f"img{i}", "caption": "a small red ball on grass"} for i in range(5)]
    (cap_dir / "captions.jsonl").write_text("\n".join(json.dumps(c) for c in caps))
    feat_dir = tmp_path / "vision"
    feat_dir.mkdir()
    # img3, img4 は特徴なし（リンク腐敗）→スキップされる想定。
    (feat_dir / "index.json").write_text(json.dumps({"rows": {"img0": 0, "img1": 1, "img2": 2}}))

    cfg = OmegaConf.create({
        "model": {"vocab_size": 300, "max_seq_len": 16},
        "data": {
            "modality": "multimodal", "word_budget": 4000, "text_caption_ratio": 0.5,
            "text_dir": str(text_dir), "caption_dir": str(cap_dir), "feature_dir": str(feat_dir),
            "processed_dir": str(tmp_path / "processed"), "tokenizer_path": str(tmp_path / "tok"),
            "backfill_ln": False,
        },
    })
    MultimodalPreprocessor(cfg).run()

    cap_meta = json.loads((mm_caption_dir(cfg.data) / "meta.json").read_text())
    assert cap_meta["n_captions"] == 3, "特徴のある img0-2 のみ採用"
    assert cap_meta["n_skipped_no_feature"] == 2, "img3,img4 は特徴なしでスキップ"
    assert cap_meta["caption_words"] <= 2000, "caption 予算（budget×(1-ratio)）内"
    # text 側も出力される。
    assert (mm_text_dir(cfg.data) / "tokens.npy").exists()
