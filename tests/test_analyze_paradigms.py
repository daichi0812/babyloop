"""per-paradigm 採点ハーネス（scripts/analyze_paradigms.py）のテスト。

一次受け入れ（最重要）: 手元の full-main BLiMP 予測を採点した全体平均が、公式
`best_temperature_report.txt` の AVERAGE ACCURACY（= ① 75.23）を再現すること。
これで「採点ロジック＋非加重平均集計」を真値に対して検証する。実データが無い環境では skip。
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_STRICT = _REPO / "third_party" / "evaluation-pipeline" / "strict"
_RESULTS = _STRICT / "results" / "babyloop-std-text"
_COLLATE = _RESULTS / "all_full_preds_and_fast_scores_causal.json"
_BLIMP_DATA = _STRICT / "evaluation_data" / "full_eval" / "blimp_filtered"
_REPORT = _RESULTS / "main" / "zero_shot" / "causal" / "blimp" / "blimp_filtered" / "best_temperature_report.txt"


def _load_module():
    path = _REPO / "scripts" / "analyze_paradigms.py"
    spec = importlib.util.spec_from_file_location("analyze_paradigms", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_paradigms"] = mod
    spec.loader.exec_module(mod)
    return mod


ap = _load_module()

_has_data = _COLLATE.exists() and _BLIMP_DATA.exists()
needs_data = pytest.mark.skipif(not _has_data, reason="手元に ① の collate 出力/BLiMP データが無い")


def _parse_report_uid(path: Path) -> dict[str, float]:
    rep, sec = {}, None
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("### UID"):
            sec = "uid"; continue
        if s.startswith("###"):
            sec = None; continue
        if sec == "uid" and ":" in s:
            k, v = s.split(":")
            rep[k.strip()] = float(v)
    return rep


def _parse_report_average(path: Path) -> float:
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("### AVERAGE"):
            return float(lines[i + 1].strip())
    raise AssertionError("AVERAGE が report に無い")


# --- 構造テスト（データ非依存に近い: マップは BLiMP データに依存） ---
@needs_data
def test_phenomenon_map_covers_67_paradigms():
    mapping = ap.load_phenomenon_map(_BLIMP_DATA)
    assert len(mapping) == 67
    # 全パラダイムが linguistics_term/field を持つ（未マップが無い）
    for st, meta in mapping.items():
        assert meta["linguistics_term"], f"{st} に linguistics_term が無い"
        assert meta["field"], f"{st} に field が無い"
    n_phenom = len({m["linguistics_term"] for m in mapping.values()})
    assert n_phenom == 13  # 実データ由来の現象数
    # field の表記ゆれ（'syntax/semantics'）が正規化され 4 種に収まる
    assert {m["field"] for m in mapping.values()} == {"syntax", "morphology", "syntax_semantics", "semantics"}


def test_revision_to_words():
    assert ap.revision_to_words("chck_1M") == 1_000_000
    assert ap.revision_to_words("chck_800M") == 800_000_000
    assert ap.revision_to_words("chck_1000M") == 1_000_000_000
    assert len(ap.FAST_REVISIONS) == 28


# --- 一次受け入れ: 75.23 再現 ---
@needs_data
def test_full_main_overall_reproduces_report_average():
    collate = json.loads(_COLLATE.read_text())
    mapping = ap.load_phenomenon_map(_BLIMP_DATA)
    full = ap.analyze_full_main(collate, mapping, _BLIMP_DATA)
    assert full["available"]
    report_avg = _parse_report_average(_REPORT)  # 75.23
    # 採点による非加重平均が公式 report の AVERAGE を ±0.1 で再現
    assert abs(full["overall"] - report_avg) < 0.1, (full["overall"], report_avg)
    assert abs(full["overall"] - 75.23) < 0.1


@needs_data
def test_per_paradigm_matches_report_uid():
    collate = json.loads(_COLLATE.read_text())
    per_paradigm = ap.grade_paradigms(collate["blimp"], _BLIMP_DATA)
    rep = _parse_report_uid(_REPORT)
    assert len(per_paradigm) == 67
    # 各パラダイムの採点値が report の UID 値と一致（temperature/丸め差のみ許容）
    max_dev = 0.0
    for st, acc in per_paradigm.items():
        rk = st if st in rep else next((r for r in rep if r.lower() == st.lower()), None)
        assert rk is not None, f"report に {st} が無い"
        max_dev = max(max_dev, abs(acc - rep[rk]))
    assert max_dev < 0.5, f"採点 vs report 乖離が大きい: {max_dev}"


@needs_data
def test_aggregation_in_range_and_complete():
    collate = json.loads(_COLLATE.read_text())
    mapping = ap.load_phenomenon_map(_BLIMP_DATA)
    full = ap.analyze_full_main(collate, mapping, _BLIMP_DATA)
    by_phenom = full["by_phenomenon"]
    assert len(by_phenom) == 13
    # n_paradigms の合計が 67（漏れも重複も無い）
    assert sum(v["n_paradigms"] for v in by_phenom.values()) == 67
    for v in by_phenom.values():
        assert 0.0 <= v["accuracy"] <= 100.0


# --- 28点軌道 ---
@needs_data
def test_trajectory_has_28_checkpoints():
    collate = json.loads(_COLLATE.read_text())
    mapping = ap.load_phenomenon_map(_BLIMP_DATA)
    traj = ap.analyze_trajectory(collate, mapping)
    revisions = {r["revision"] for r in traj}
    assert len(revisions) == 28
    assert revisions == set(ap.FAST_REVISIONS)
    # 各 revision は 13 現象 + __overall__ = 14 行
    for rev in ap.FAST_REVISIONS:
        n = sum(1 for r in traj if r["revision"] == rev)
        assert n == 14, (rev, n)
    for r in traj:
        assert 0.0 <= r["accuracy"] <= 100.0
        assert r["words"] == ap.revision_to_words(r["revision"])


@needs_data
def test_trajectory_index_alignment_matches_full_main():
    """index→revision→words の整合 ground-truth: 最終 chck_1000M の fast overall が
    full-main overall(75.23) に一致するはず（≈、200件fast vs full の差のみ許容）。
    ordering がずれると構造テストは通るが科学が壊れる。その唯一の盲点を塞ぐ。"""
    collate = json.loads(_COLLATE.read_text())
    mapping = ap.load_phenomenon_map(_BLIMP_DATA)
    traj = ap.analyze_trajectory(collate, mapping)
    full = ap.analyze_full_main(collate, mapping, _BLIMP_DATA)
    final = [r for r in traj if r["revision"] == "chck_1000M" and r["phenomenon"] == "__overall__"]
    assert len(final) == 1
    # fast(200件) と full の差は小さいはず。大きくずれたら index 整合が壊れている。
    assert abs(final[0]["accuracy"] - full["overall"]) < 3.0, (final[0]["accuracy"], full["overall"])


# --- 統計報告（効果量 + CI）: ③ の 3seed を模した sanity ---
def test_effect_size_and_ci_directions():
    # ③ loop_text(76.61/75.71/76.17) vs ① std_text(75.22/75.88/74.53)
    loop = [76.61, 75.71, 76.17]
    std = [75.22, 75.88, 74.53]
    res = ap.mean_diff_ci(loop, std)
    assert res["diff"] > 0  # ③ > ①
    assert res["cohens_d"] > 0
    # n=3 で非有意（p≈0.1）= CI が 0 を跨ぐ想定
    assert res["ci_low"] < 0 < res["ci_high"]


def test_cohens_d_zero_for_identical_groups():
    import math
    d = ap.cohens_d([1.0, 1.0], [1.0, 1.0])
    assert math.isnan(d)  # pooled SD=0 → 定義不能(nan)
