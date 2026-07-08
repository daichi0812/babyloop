"""words_seen 駆動 LR スケジュールの不変条件。

LR 地平線を max_steps（セル毎にバッチ構成で変わる）から max_words_seen（全セル共通）へ移した
（trainer.cosine_lr_words）。ロックする不変条件:
  1. packed text（1 step≒一定語数）では words 駆動 LR が旧 step 駆動 LR を再現＝①③ は no-op
     （再ラン不要。可変長 caption の MM だけ正しく直る、という主張の土台）。
  2. warmup 起点≈0 / 地平線で min_lr / post-warmup 単調減少。
"""

import math

from babyloop.training.trainer import cosine_lr_words


def _step_lr(step, peak, min_ratio, warmup_steps, max_steps):
    """移行前の step 駆動 cosine（参照実装）。"""
    if step < warmup_steps:
        return peak * (step / max(1, warmup_steps))
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    return peak * (min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress)))


def test_words_driven_matches_step_driven_for_packed_text():
    """packed text（一定 語/step）では words 駆動 == step 駆動（①③ 再現＝no-op）。"""
    peak, min_ratio = 1.5e-3, 0.1
    max_steps, warmup_steps = 6827, 102      # ①本番相当
    wps = 146_000                            # packed text の一定 語/step（定数なら値は任意）
    max_words = max_steps * wps
    warmup_frac = warmup_steps / max_steps   # 語数比 = step 比（packed なので一致）
    warmup_words = warmup_frac * max_words
    for s in range(0, max_steps + 1, 50):
        a = _step_lr(s, peak, min_ratio, warmup_steps, max_steps)
        b = cosine_lr_words(s * wps, peak, min_ratio, warmup_words, max_words)
        assert abs(a - b) < 1e-9, f"step={s}: step={a} words={b}"


def test_words_driven_endpoints_and_monotonic():
    peak, min_ratio, max_words = 1.5e-3, 0.1, 1_000_000_000
    warmup_words = 0.015 * max_words
    assert cosine_lr_words(0, peak, min_ratio, warmup_words, max_words) == 0.0
    assert abs(cosine_lr_words(warmup_words, peak, min_ratio, warmup_words, max_words) - peak) < 1e-12
    assert abs(cosine_lr_words(max_words, peak, min_ratio, warmup_words, max_words) - peak * min_ratio) < 1e-12
    prev = peak + 1
    for w in range(int(warmup_words), max_words + 1, max_words // 20):
        cur = cosine_lr_words(w, peak, min_ratio, warmup_words, max_words)
        assert cur <= prev + 1e-12  # post-warmup は単調減少
        prev = cur


def test_horizon_is_words_not_steps():
    """地平線が max_words_seen＝同じ words_seen なら max_steps に依らず同じ LR（MM/text 共通の形）。"""
    peak, min_ratio, max_words = 1.5e-3, 0.1, 1_000_000_000
    warmup_words = 0.015 * max_words
    # 同一 words_seen=5e8 での LR は、何 step で到達したか（=max_steps の取り方）に依存しない。
    lr_at_half = cosine_lr_words(5e8, peak, min_ratio, warmup_words, max_words)
    assert abs(lr_at_half - peak * (min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * (5e8 - warmup_words) / (max_words - warmup_words))))) < 1e-12
