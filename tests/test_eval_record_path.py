"""eval 記録先を checkpoint パスから導く `run_id_from_checkpoint` のテスト。

背景: 以前 `scripts/evaluate.py` は eval.json の保存先を `cfg.train.seed`（default 42）で
決めていたため、`+checkpoint=...seed_43...` を渡しても `train.seed` を上書きし忘れると
seed_42 の記録をサイレント上書きした。記録先を checkpoint パス由来に固定して再発を防ぐ。
"""

import pytest

from babyloop.training.run_record import run_id_from_checkpoint


@pytest.mark.parametrize(
    "path, expected",
    [
        ("outputs/loop_mm/seed_43/ckpt_final", ("loop_mm", 43)),
        ("outputs/std_text/seed_42/ckpt_final", ("std_text", 42)),
        ("outputs/loop_text/seed_44/ckpt_final", ("loop_text", 44)),
        # 絶対パス（A100 実行時の形）
        ("/home/hotta/dev-hub/research/ml/babylm/babyloop/outputs/loop_mm/seed_43/ckpt_final",
         ("loop_mm", 43)),
        # 末尾が milestone でも seed を拾う
        ("outputs/std_text/seed_44/chck_500M", ("std_text", 44)),
        # seed dir 自体を指す
        ("outputs/loop_mm/seed_7", ("loop_mm", 7)),
    ],
)
def test_run_id_from_checkpoint_parses(path, expected):
    assert run_id_from_checkpoint(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "some/random/path/model",          # seed_<N> セグメント無し
        "outputs/std_text/ckpt_final",     # seed dir 無し
        "outputs/std_text/seedXX/ckpt",    # seed_<数字> でない
        "",
    ],
)
def test_run_id_from_checkpoint_none_when_no_pattern(path):
    assert run_id_from_checkpoint(path) is None


def test_seed43_does_not_resolve_to_seed42():
    """今回のバグの回帰テスト: seed_43 の checkpoint が seed_42 に解決されないこと。"""
    name, seed = run_id_from_checkpoint("outputs/loop_mm/seed_43/ckpt_final")
    assert (name, seed) == ("loop_mm", 43)
    assert seed != 42
