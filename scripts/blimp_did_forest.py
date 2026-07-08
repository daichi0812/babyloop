"""BLiMP 現象別 DiD フォレストプロット（3seed・論文図）。

集計値の DiD null（recurrence×grounding の交互作用がゼロ）が「現象間で効果が
相殺しているだけの見かけのゼロ」でないことを示すための**探索的**分解図。
公式 eval-pipeline の各 report.txt に既にある `### LINGUISTICS_TERM ACCURACY`
（13 現象）を全条件×3seed からパースし、現象別に DiD を per-seed で計算する。

推定量は 2 系列（両者とも第2括弧 (std_mm − std_text) は共通）:
  - matched     : (loop_mm_prefix − loop_text) − (std_mm − std_text)   … 主推定量（injection-matched, headline −0.01）
  - re-injected : (loop_mm        − loop_text) − (std_mm − std_text)   … 副推定量（N_inj=2, +0.06）

per-seed DiD は 4 条件を seed 番号でペアリングする（独立初期化なので本質的に任意
＝paper §per-seed の caveat と同じ）。図では平均マーカー＋3seed 個別点＋min–max
ひげのみを描き、n=3 を超える確度を示唆する CI は引かない。

出力:
  - analysis/outputs/per_phenomenon_did_3seed.csv （元データ）
  - papers/babylm_challenge/figs/fig_blimp_did.pdf / .svg （ベクタ・英語のみ）

使い方:
    uv run --group analysis python scripts/blimp_did_forest.py
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_RESULTS = _REPO / "outputs" / "eval_pipeline" / "strict_results"
_REPORT_REL = "main/zero_shot/causal/blimp/blimp_filtered/best_temperature_report.txt"

SEEDS = [42, 43, 44]

# 条件 → {seed: 結果ディレクトリ}。無印 dir は leaderboard 提出 seed（std/loop_text/std_mm=42、
# loop_mm のみ 43。paper §Eval 脚注）。AVERAGE と paper per-seed 表の突き合わせで検証する（下記）。
COND_DIRS: dict[str, dict[int, str]] = {
    "std_text": {42: "babyloop-std-text", 43: "babyloop-std-text-s43", 44: "babyloop-std-text-s44"},
    "loop_text": {42: "babyloop-loop-text", 43: "babyloop-loop-text-s43", 44: "babyloop-loop-text-s44"},
    "std_mm": {42: "babyloop-std-mm", 43: "babyloop-std-mm-s43", 44: "babyloop-std-mm-s44"},
    "loop_mm": {42: "babyloop-loop-mm-s42", 43: "babyloop-loop-mm", 44: "babyloop-loop-mm-s44"},
    "loop_mm_prefix": {
        42: "babyloop-loop-mm-prefix-s42",
        43: "babyloop-loop-mm-prefix-s43",
        44: "babyloop-loop-mm-prefix-s44",
    },
}

# paper per-seed 表（tab:perseed, main.tex）の BLiMP full AVERAGE。seed→dir 対応の検証用。
EXPECTED_AVG: dict[str, dict[int, float]] = {
    "std_text": {42: 75.23, 43: 75.88, 44: 74.53},
    "loop_text": {42: 76.61, 43: 75.71, 44: 76.17},
    "std_mm": {42: 71.62, 43: 71.05, 44: 71.44},
    "loop_mm": {42: 71.56, 43: 73.21, 44: 72.36},
    "loop_mm_prefix": {42: 72.34, 43: 71.66, 44: 72.92},
}

# BLiMP 標準カテゴリ順（効果量ソートではなく固定）。表示ラベルは英語・簡潔化。
# s-selection は正準 12 現象には無い linguistics_term なので、意味的近縁の argument_structure 直後に置く。
PHENOM_ORDER: list[tuple[str, str]] = [
    ("anaphor_agreement", "Anaphor agr."),
    ("argument_structure", "Argument structure"),
    ("s-selection", "S-selection"),
    ("binding", "Binding"),
    ("control_raising", "Control/raising"),
    ("determiner_noun_agreement", "Det.-noun agr."),
    ("ellipsis", "Ellipsis"),
    ("filler_gap_dependency", "Filler-gap"),
    ("irregular_forms", "Irregular forms"),
    ("island_effects", "Island effects"),
    ("npi_licensing", "NPI licensing"),
    ("quantifiers", "Quantifiers"),
    ("subject_verb_agreement", "Subject-verb agr."),
]


def parse_report(path: Path) -> tuple[dict[str, float], float]:
    """report.txt から (linguistics_term→acc, AVERAGE) を返す。"""
    section = None
    phenom: dict[str, float] = {}
    average = float("nan")
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("### "):
            section = line[4:].strip()
            continue
        if not line:
            continue
        if section == "LINGUISTICS_TERM ACCURACY":
            key, val = line.split(":", 1)
            phenom[key.strip()] = float(val)
        elif section == "AVERAGE ACCURACY":
            average = float(line)
    return phenom, average


def load_all() -> dict[str, dict[int, dict[str, float]]]:
    """{cond: {seed: {phenom: acc}}} を読み、seed→dir 対応を AVERAGE で検証する。"""
    data: dict[str, dict[int, dict[str, float]]] = {}
    for cond, dirs in COND_DIRS.items():
        data[cond] = {}
        for seed, d in dirs.items():
            path = _RESULTS / d / _REPORT_REL
            if not path.exists():
                raise FileNotFoundError(f"report 不在: {path}")
            phenom, avg = parse_report(path)
            exp = EXPECTED_AVG[cond][seed]
            if abs(avg - exp) > 0.05:
                raise AssertionError(
                    f"seed→dir 不整合の疑い: {cond} s{seed} ({d}) AVERAGE={avg} != 期待 {exp}"
                )
            data[cond][seed] = phenom
    return data


def cross_check_phenomenon_parse(data: dict[str, dict[int, dict[str, float]]]) -> None:
    """LINGUISTICS_TERM パースを別ハーネス由来の per_phenomenon.csv（n=1・std_text seed42）と突き合わせる。

    AVERAGE assert は現象別値を担保しないので、collate 経由（temp=1）の既存 csv と ±1.0 以内で
    一致することを確認し、report パース自体の正しさを閉じる。csv 不在なら skip。
    """
    csv_path = _REPO / "analysis" / "outputs" / "per_phenomenon.csv"
    if not csv_path.exists():
        print("[did] cross-check skip（per_phenomenon.csv 不在）")
        return
    ref: dict[str, float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["phenomenon"] != "__overall__":
                ref[row["phenomenon"]] = float(row["std_text"])
    got = data["std_text"][42]
    worst = 0.0
    for key, _label in PHENOM_ORDER:
        if key in ref and key in got:
            worst = max(worst, abs(ref[key] - got[key]))
    if worst > 1.0:  # temperature 差でも通常 ~0.5 以内
        raise AssertionError(f"現象別パース不一致の疑い: std_text seed42 max|Δ|={worst:.2f} > 1.0")
    print(f"[did] cross-check OK: std_text 現象別 max|Δ| vs per_phenomenon.csv = {worst:.2f}")


def compute_did(data: dict[str, dict[int, dict[str, float]]]) -> dict[str, dict[str, list[float]]]:
    """現象別 per-seed DiD を matched / reinjected の 2 系列で返す: {estimand: {phenom: [seed別 DiD]}}."""
    estimands = {"matched": "loop_mm_prefix", "reinjected": "loop_mm"}
    out: dict[str, dict[str, list[float]]] = {}
    for name, loop_mm_cond in estimands.items():
        out[name] = {}
        for key, _label in PHENOM_ORDER:
            vals: list[float] = []
            for s in SEEDS:
                a = data[loop_mm_cond][s][key] - data["loop_text"][s][key]
                b = data["std_mm"][s][key] - data["std_text"][s][key]
                vals.append(a - b)
            out[name][key] = vals
    return out


def _sanity_overall(data: dict[str, dict[int, dict[str, float]]]) -> None:
    """全体 DiD（AVERAGE ベース）が paper headline を再現するか確認して print。"""
    for name, cond in (("matched", "loop_mm_prefix"), ("reinjected", "loop_mm")):
        per_seed = []
        for s in SEEDS:
            a = EXPECTED_AVG[cond][s] - EXPECTED_AVG["loop_text"][s]
            b = EXPECTED_AVG["std_mm"][s] - EXPECTED_AVG["std_text"][s]
            per_seed.append(a - b)
        print(
            f"[did] overall {name}: per-seed={[round(v, 2) for v in per_seed]} "
            f"mean={statistics.mean(per_seed):+.2f}"
        )


def write_csv(did: dict[str, dict[str, list[float]]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "per_phenomenon_did_3seed.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["estimand", "phenomenon", "did_s42", "did_s43", "did_s44", "did_mean", "did_sd", "did_min", "did_max"]
        )
        for name, per_phenom in did.items():
            for key, _label in PHENOM_ORDER:
                v = per_phenom[key]
                w.writerow(
                    [name, key, f"{v[0]:.2f}", f"{v[1]:.2f}", f"{v[2]:.2f}",
                     f"{statistics.mean(v):.2f}", f"{statistics.pstdev(v):.2f}",
                     f"{min(v):.2f}", f"{max(v):.2f}"]
                )
    print(f"[did] csv -> {path}")
    return path


def plot_forest(did: dict[str, dict[str, list[float]]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42})

    labels = [lbl for _key, lbl in PHENOM_ORDER]
    n = len(PHENOM_ORDER)
    y = list(range(n))  # 0..n-1; 反転して先頭を上に

    # matched=主（塗り丸・濃色）, reinjected=副（白抜き四角・橙）。行内で上下オフセット。
    series = [
        ("matched", +0.16, "#1f3b73", "o", True, "matched (injection-matched, primary)"),
        ("reinjected", -0.16, "#d1701a", "s", False, "re-injected ($N_{\\mathrm{inj}}{=}2$)"),
    ]

    fig, ax = plt.subplots(figsize=(3.35, 3.8))

    # 交互の薄い横帯（可読性）
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="0.94", zorder=0)
    ax.axvline(0.0, color="0.4", lw=0.9, ls="--", zorder=1)

    for name, dy, color, marker, filled, _legend in series:
        for i, (key, _lbl) in enumerate(PHENOM_ORDER):
            vals = did[name][key]
            yy = y[i] + dy
            mean = statistics.mean(vals)
            # min–max ひげ
            ax.plot([min(vals), max(vals)], [yy, yy], color=color, lw=0.8, alpha=0.55, zorder=2)
            # 個別 seed 点
            ax.scatter(vals, [yy] * len(vals), s=7, color=color, alpha=0.45,
                       edgecolors="none", zorder=3)
            # 平均マーカー
            ax.scatter([mean], [yy], s=26, marker=marker,
                       facecolors=(color if filled else "white"),
                       edgecolors=color, linewidths=1.1, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("per-phenomenon DiD (BLiMP acc. pts)")
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", color="#1f3b73", markerfacecolor="#1f3b73",
               markersize=6, lw=0, label="matched (primary)"),
        Line2D([0], [0], marker="s", color="#d1701a", markerfacecolor="white",
               markersize=6, lw=0, label="re-injected"),
        Line2D([0], [0], marker="o", color="0.5", markerfacecolor="0.5",
               markersize=3.5, lw=0, alpha=0.6, label="individual seeds"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=3, fontsize=6.2, frameon=False, handletextpad=0.3,
              columnspacing=1.0, borderpad=0.2)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "svg"):
        p = out_dir / f"fig_blimp_did.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"[did] figure -> {p}")
    plt.close(fig)


def main() -> None:
    data = load_all()
    cross_check_phenomenon_parse(data)
    _sanity_overall(data)
    did = compute_did(data)
    write_csv(did, _REPO / "analysis" / "outputs")
    plot_forest(did, _REPO / "papers" / "babylm_challenge" / "figs")


if __name__ == "__main__":
    main()
