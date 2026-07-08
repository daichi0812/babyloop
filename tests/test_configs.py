"""4つのexperiment configが2×2表どおりに合成されることを検証する。"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# 2×2デザイン: (アーキテクチャ, モダリティ) → 期待値
CELLS = {
    "std_text": {"k": 1, "modality": "text"},
    "std_mm": {"k": 1, "modality": "multimodal"},
    "loop_text": {"k": 4, "modality": "text"},
    "loop_mm": {"k": 4, "modality": "multimodal"},
}


@pytest.mark.parametrize("experiment", CELLS)
def test_experiment_composition(experiment):
    expected = CELLS[experiment]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

    assert cfg.name == experiment
    assert cfg.model.k == expected["k"]
    assert cfg.data.modality == expected["modality"]


@pytest.mark.parametrize("experiment", ["std_mm", "loop_mm"])
def test_multimodal_cells_enable_vision(experiment):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

    assert cfg.model.vision_encoder.enabled
    assert cfg.model.fusion is not None


@pytest.mark.parametrize("experiment", ["std_text", "loop_text"])
def test_text_cells_disable_vision(experiment):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

    assert not cfg.model.vision_encoder.enabled
    assert cfg.model.fusion is None


# ADR-0006: 融合の軸＝注入場所（connector.inject_iters）＋ projector 型（connector.type）。
# fusion は視覚 on/off ゲート専用（型は持たない）。naive 劣化は connector.type=identity 一本。
@pytest.mark.parametrize(
    "experiment,ctype,inject",
    [("std_mm", "mlp", 0), ("std_mm_naive", "identity", 0)],
)
def test_fusion_axis_semantics(experiment, ctype, inject):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])
    assert cfg.model.fusion is not None, "fusion は視覚 on ゲート（not None）"
    assert cfg.model.connector.type == ctype, "projector 型は connector.type"
    assert cfg.model.connector.inject_iters == inject


def test_prefix_vs_inloop_injection_site():
    """②(std_mm)=prefix(inject_iters==0) / ④(loop_mm)=in-loop(inject_iters>0)。"""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        std = compose(config_name="config", overrides=["experiment=std_mm"])
        loop = compose(config_name="config", overrides=["experiment=loop_mm"])
    assert std.model.connector.inject_iters == 0
    assert loop.model.connector.inject_iters > 0


def test_loop_mm_injection_mode_and_clean_baseline():
    """④ loop_mm は in-loop 注入(inject_iters>0)・既定 mode=prefix_refresh(B)。
    clean baseline loop_mm_prefix は looped×視覚 prefix で再注入なし(inject_iters==0)。"""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        loop = compose(config_name="config", overrides=["experiment=loop_mm"])
        clean = compose(config_name="config", overrides=["experiment=loop_mm_prefix"])
    assert loop.model.connector.inject_iters > 0
    assert loop.model.connector.inject_mode == "prefix_refresh"  # B 主系
    # clean baseline: ループ(k>1)＋視覚 on＋再注入なし。
    assert clean.model.k > 1
    assert clean.model.fusion is not None
    assert clean.model.connector.inject_iters == 0


# docs/design.md が謳う構造と config が乖離したら落ちる回帰テスト。
# チューニング値（lr, batch 等）はスイープで動くのでアサートしない。設計の不変条件だけをロックする。
@pytest.mark.parametrize("experiment", CELLS)
def test_architecture_invariants(experiment):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])
    m = cfg.model

    assert m.d_model % m.n_heads == 0, "d_model はヘッド数で割り切れること"
    assert m.d_model // m.n_heads == 64, "head_dim = 64（設計確定値）"
    assert m.n_layers == m.n_prelude + m.n_core + m.n_coda, "n_layers = prelude + core + coda"
    assert m.tie_embeddings is True, "embedding と lm_head は tie"
    assert m.bias is False, "Linear は bias 無し"


def test_standard_degenerates_to_k1():
    """標準セルは K=1（LoopedTransformer が標準に縮退する前提、ADR-0001）。"""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=std_text"])
    assert cfg.model.k == 1
    assert cfg.model.inject_input is False
