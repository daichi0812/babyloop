"""checkpoint マイルストーン命名・スケジュールのテスト。

提出規定（公式 collate_preds.py の ``OTHER_FAST_REVISIONS`` = ``chck_*M`` 28点）
からズレると提出が無効になるため、命名と28点スケジュールをロックする。
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from babyloop.training.checkpointing import milestone_label, official_strict_milestones

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# 公式 collate_preds.py の OTHER_FAST_REVISIONS（Strict）と同一の生成規則。
EXPECTED_LABELS = (
    [f"chck_{i}M" for i in range(1, 10)]
    + [f"chck_{i * 10}M" for i in range(1, 10)]
    + [f"chck_{i * 100}M" for i in range(1, 11)]
)


def test_milestone_label_uses_M_units_no_billion():
    """1000M は '1B' でなく '1000M'（collate は M単位を仮定）。"""
    assert milestone_label(1_000_000) == "1M"
    assert milestone_label(100_000_000) == "100M"
    assert milestone_label(1_000_000_000) == "1000M"
    assert milestone_label(10_000) == "10K"  # 疎通用（提出対象外）


def test_official_schedule_matches_collate_revisions():
    """official_strict_milestones() の chck_ ラベルが公式28点と完全一致すること。"""
    milestones = official_strict_milestones()
    assert len(milestones) == 28
    labels = [f"chck_{milestone_label(m)}" for m in milestones]
    assert labels == EXPECTED_LABELS


def test_default_train_config_uses_official_schedule():
    """本番 config（std_text）の checkpoint_milestones が公式28点であること。"""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=std_text"])
    assert list(cfg.train.checkpoint_milestones) == official_strict_milestones()


def test_smoke_config_overrides_milestones():
    """疎通 config はデバッグ用の小マイルストーンに上書きされること（提出対象外）。"""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=std_text_smoke"])
    assert list(cfg.train.checkpoint_milestones) == [10000, 20000]
    assert int(cfg.train.grad_accum_steps) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
