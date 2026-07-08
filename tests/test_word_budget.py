"""WordBudgetTracker のテスト。

word budget計上が壊れると提出が無効になるため、K=1等価性と並ぶ
クリティカルなテスト。
"""

import pytest

from babyloop.data.word_budget import BudgetExceededError, WordBudgetTracker


def test_count_words_follows_official_rule():
    """語数カウントがBabyLM公式ルール（空白区切り）と一致すること。"""
    assert WordBudgetTracker.count_words("the cat sat") == 3
    assert WordBudgetTracker.count_words("  multiple   spaces \t and\nnewlines ") == 4
    assert WordBudgetTracker.count_words("") == 0
    assert WordBudgetTracker.count_words("one") == 1


def test_budget_exceeded_raises():
    """budget超過時に BudgetExceededError が送出されること。"""
    tracker = WordBudgetTracker(budget=10)
    tracker.add(6)
    assert tracker.remaining == 4
    tracker.add(4)
    assert tracker.remaining == 0
    with pytest.raises(BudgetExceededError):
        tracker.add(1)


def test_add_accumulates_via_count_words():
    """count_words の結果をそのまま計上できること（前処理での使い方）。"""
    tracker = WordBudgetTracker(budget=100)
    for line in ["a b c", "d e", "f"]:
        tracker.add(WordBudgetTracker.count_words(line))
    assert tracker.consumed == 6
    assert tracker.remaining == 94


def test_caption_words_count_toward_budget(tmp_path):
    """マルチモーダルtrackでキャプション語数がbudgetに計上され、予算で打ち切られること。

    word_budget を text_caption_ratio で分け、caption 予算（budget×(1-ratio)）を超える分は
    打ち切る（既定: 得られた語数で締める）。caption 語数が予算に計上される＝提出規定の遵守。
    """
    import json

    from omegaconf import OmegaConf

    from babyloop.data.preprocessing import MultimodalPreprocessor, mm_caption_dir

    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "train.txt").write_text(
        "\n".join("the quick brown fox jumps over the lazy dog" for _ in range(40))
    )
    cap_dir = tmp_path / "captions"
    cap_dir.mkdir()
    # 50 captions × 6 語 = 300 語ぶん。caption 予算 100 語に対し超過する。
    caps = [{"image_id": f"img{i}", "caption": "a small red ball on grass"} for i in range(50)]
    (cap_dir / "captions.jsonl").write_text("\n".join(json.dumps(c) for c in caps))
    feat_dir = tmp_path / "vision"
    feat_dir.mkdir()
    (feat_dir / "index.json").write_text(json.dumps({"rows": {f"img{i}": i for i in range(50)}}))

    cfg = OmegaConf.create({
        "model": {"vocab_size": 300, "max_seq_len": 16},
        "data": {
            "modality": "multimodal", "word_budget": 200, "text_caption_ratio": 0.5,
            "text_dir": str(text_dir), "caption_dir": str(cap_dir), "feature_dir": str(feat_dir),
            "processed_dir": str(tmp_path / "processed"), "tokenizer_path": str(tmp_path / "tok"),
            "backfill_ln": False,
        },
    })
    MultimodalPreprocessor(cfg).run()

    cap_meta = json.loads((mm_caption_dir(cfg.data) / "meta.json").read_text())
    caption_budget = 200 * (1 - 0.5)  # = 100
    assert cap_meta["caption_words"] > 0, "caption 語数が計上されること"
    assert cap_meta["caption_words"] <= caption_budget, "caption 予算で打ち切られること"
    assert cap_meta["n_captions"] < 50, "予算超過分の caption は採用されない"
