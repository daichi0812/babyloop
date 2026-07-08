"""BLiMP を現象レベルに分解する分析ハーネス（collate 出力を入力に、GPU 不要）。

公式 collate 出力 `all_full_preds_and_fast_scores_<backend>.json` を読み、
BLiMP の 67 パラダイム（UID）を **現象カテゴリ**（`linguistics_term`=13種 / `field`=4種）に集約する。
スカラー値（例: ① 75.21）の脆さに対し、現象レベルの分解と words_seen 軌道で頑健な所見を出すのが狙い。

出力は 2 系統:
  1. full-main per-phenomenon: 最終モデルの BLiMP full 予測を採点 → 現象別正答率の表。
     採点は公式 `collate_preds._calculate_target_results` と同一ロジック（pred==sentence_good）で、
     非加重平均（公式 print_results_table 準拠）が report の AVERAGE ACCURACY を再現する。
  2. 28点 words_seen 軌道 per-phenomenon: `fast_eval_results['blimp']`（採点済み）を現象集約し、
     どの現象が早期飽和し、どれが 1000M まで伸びるかを可視化。

複数条件（①text / ②④MM 等）を同時に渡せば横並び＋差分を出す（②④の collate 到着後に同コマンドで比較可能）。

使い方:
    uv run --group analysis python scripts/analyze_paradigms.py \
        --collate std_text=third_party/evaluation-pipeline/strict/results/babyloop-std-text/all_full_preds_and_fast_scores_causal.json
    # 複数条件（--collate を繰り返す）:
    uv run --group analysis python scripts/analyze_paradigms.py \
        --collate text=<...causal.json> --collate loop_mm=<...causal.json> --out analysis/outputs
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_STRICT = _REPO / "third_party" / "evaluation-pipeline" / "strict"
DEFAULT_BLIMP_DATA = _STRICT / "evaluation_data" / "full_eval" / "blimp_filtered"

# collate の OTHER_FAST_REVISIONS と同一順序（fast_eval_results の list はこの順で append される）。
FAST_REVISIONS: list[str] = (
    [f"chck_{i}M" for i in range(1, 10)]
    + [f"chck_{i * 10}M" for i in range(1, 10)]
    + [f"chck_{i * 100}M" for i in range(1, 11)]
)


def revision_to_words(revision: str) -> int:
    """'chck_800M' -> 800_000_000（words 駆動スケジュールの words_seen 地平に対応）。"""
    return int(revision.removeprefix("chck_").removesuffix("M")) * 1_000_000


def _normalize_field(field: str | None) -> str | None:
    """field の表記ゆれ（'syntax/semantics' と 'syntax_semantics'）を統一。"""
    if field is None:
        return None
    return field.replace("/", "_")


def load_phenomenon_map(data_dir: Path = DEFAULT_BLIMP_DATA) -> dict[str, dict[str, str]]:
    """blimp データ jsonl 群から subtask(UID) -> {linguistics_term, field} を構築（ハードコードしない）。"""
    mapping: dict[str, dict[str, str]] = {}
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"BLiMP データ jsonl が見つかりません: {data_dir}")
    for path in files:
        with path.open() as f:
            row = json.loads(f.readline())
        mapping[path.stem] = {
            "linguistics_term": row.get("linguistics_term"),
            "field": _normalize_field(row.get("field")),
        }
    return mapping


def _resolve(name: str, mapping: dict[str, dict[str, str]]) -> dict[str, str] | None:
    """subtask 名を mapping に解決（大文字小文字の差を吸収）。"""
    if name in mapping:
        return mapping[name]
    lower = {k.lower(): v for k, v in mapping.items()}
    return lower.get(name.lower())


def grade_paradigms(
    blimp_predictions: dict[str, dict], data_dir: Path = DEFAULT_BLIMP_DATA
) -> dict[str, float]:
    """full-main の BLiMP 生予測を採点 → {subtask: accuracy(0-100)}。

    公式 `_calculate_target_results` と同一: pred.strip() == sentence_good.strip()。
    """
    data_files = {p.stem: p for p in data_dir.glob("*.jsonl")}
    data_lower = {k.lower(): k for k in data_files}
    accs: dict[str, float] = {}
    for subtask, payload in blimp_predictions.items():
        stem = subtask if subtask in data_files else data_lower.get(subtask.lower())
        if stem is None:
            raise KeyError(f"採点データが見つからない subtask: {subtask}")
        lines = data_files[stem].read_text().splitlines()
        preds = payload["predictions"]
        correct = total = 0
        for pred, line in zip(preds, lines):
            data = json.loads(line)
            res = pred["pred"].strip() if isinstance(pred["pred"], str) else pred["pred"]
            target = data["sentence_good"].strip()
            total += 1
            if res == target:
                correct += 1
        accs[subtask] = 100.0 * correct / total
    return accs


def aggregate_by_phenomenon(
    per_paradigm: dict[str, float],
    mapping: dict[str, dict[str, str]],
    key: str = "linguistics_term",
) -> dict[str, dict]:
    """パラダイム別 accuracy を現象カテゴリに非加重平均で集約。

    返り値: {phenomenon: {"accuracy": mean, "n_paradigms": k, "paradigms": [...]}}。
    """
    buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for subtask, acc in per_paradigm.items():
        meta = _resolve(subtask, mapping)
        cat = meta.get(key) if meta else None
        if cat is None:
            raise KeyError(f"{subtask} が {key} にマップできません（mapping 欠損）")
        buckets[cat].append((subtask, acc))
    out: dict[str, dict] = {}
    for cat, items in buckets.items():
        out[cat] = {
            "accuracy": sum(a for _, a in items) / len(items),
            "n_paradigms": len(items),
            "paradigms": sorted(s for s, _ in items),
        }
    return out


def overall_accuracy(per_paradigm: dict[str, float]) -> float:
    """67 パラダイムの非加重平均（= report の AVERAGE ACCURACY 定義）。"""
    return sum(per_paradigm.values()) / len(per_paradigm)


def analyze_full_main(
    collate: dict, mapping: dict[str, dict[str, str]], data_dir: Path = DEFAULT_BLIMP_DATA
) -> dict:
    """full-main BLiMP の per-paradigm / per-phenomenon(13) / per-field(4) / overall を返す。"""
    blimp = collate.get("blimp")
    if not blimp:
        return {"available": False}
    per_paradigm = grade_paradigms(blimp, data_dir)
    return {
        "available": True,
        "overall": overall_accuracy(per_paradigm),
        "per_paradigm": per_paradigm,
        "by_phenomenon": aggregate_by_phenomenon(per_paradigm, mapping, "linguistics_term"),
        "by_field": aggregate_by_phenomenon(per_paradigm, mapping, "field"),
    }


def analyze_trajectory(collate: dict, mapping: dict[str, dict[str, str]]) -> list[dict]:
    """28 点 fast の per-phenomenon 軌道を tidy 行で返す。

    `fast_eval_results['blimp']` は採点済み（{subtask: acc 0-1}）×revision。
    返り値の各行: {revision, words, phenomenon, accuracy(0-100), n_paradigms}。
    """
    fast = collate.get("fast_eval_results", {}).get("blimp")
    if not fast:
        return []
    rows: list[dict] = []
    for i, rev_scores in enumerate(fast):
        if rev_scores is None:
            continue
        revision = FAST_REVISIONS[i] if i < len(FAST_REVISIONS) else f"idx_{i}"
        # acc は 0-1 で入っているので 100 倍して採点ロジックと単位を揃える。
        per_paradigm = {st: v * 100.0 for st, v in rev_scores.items()}
        by_phenom = aggregate_by_phenomenon(per_paradigm, mapping, "linguistics_term")
        words = revision_to_words(revision) if revision.startswith("chck_") else None
        for phenom, info in by_phenom.items():
            rows.append(
                {
                    "revision": revision,
                    "words": words,
                    "phenomenon": phenom,
                    "accuracy": info["accuracy"],
                    "n_paradigms": info["n_paradigms"],
                }
            )
        # overall 行も足す（全体軌道の確認用）。
        rows.append(
            {
                "revision": revision,
                "words": words,
                "phenomenon": "__overall__",
                "accuracy": overall_accuracy(per_paradigm),
                "n_paradigms": len(per_paradigm),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 統計報告（効果量 + 信頼区間）: n=3 で p≈0.1 だった件への対処。p 値単独にしない。
# ---------------------------------------------------------------------------
def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """対応なし 2 群の Cohen's d（pooled SD）。a - b の符号。"""
    na, nb = len(group_a), len(group_b)
    ma = sum(group_a) / na
    mb = sum(group_b) / nb
    va = sum((x - ma) ** 2 for x in group_a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in group_b) / (nb - 1) if nb > 1 else 0.0
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (ma - mb) / pooled if pooled > 0 else float("nan")


def mean_diff_ci(group_a: list[float], group_b: list[float], conf: float = 0.95) -> dict:
    """平均差 (a-b) と Welch t による信頼区間・効果量をまとめて返す。

    scipy があれば t 分布、なければ正規近似。seed×条件のスコアが揃った段階で使う想定。
    """
    na, nb = len(group_a), len(group_b)
    ma, mb = sum(group_a) / na, sum(group_b) / nb
    va = sum((x - ma) ** 2 for x in group_a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in group_b) / (nb - 1) if nb > 1 else 0.0
    se = math.sqrt(va / na + vb / nb)
    diff = ma - mb
    # Welch–Satterthwaite df
    df = (va / na + vb / nb) ** 2 / (
        ((va / na) ** 2 / (na - 1) if na > 1 else 0.0)
        + ((vb / nb) ** 2 / (nb - 1) if nb > 1 else 0.0)
    ) if se > 0 else float("nan")
    try:
        from scipy import stats  # type: ignore

        tcrit = stats.t.ppf(1 - (1 - conf) / 2, df) if se > 0 else float("nan")
        tstat = diff / se if se > 0 else float("nan")
        pval = 2 * stats.t.sf(abs(tstat), df) if se > 0 else float("nan")
    except Exception:  # scipy 不在時の正規近似フォールバック
        tcrit = 1.96
        tstat = diff / se if se > 0 else float("nan")
        pval = float("nan")
    return {
        "mean_a": ma,
        "mean_b": mb,
        "diff": diff,
        "ci_low": diff - tcrit * se if se > 0 else float("nan"),
        "ci_high": diff + tcrit * se if se > 0 else float("nan"),
        "cohens_d": cohens_d(group_a, group_b),
        "t": tstat,
        "df": df,
        "p_value": pval,
        "conf": conf,
    }


# ---------------------------------------------------------------------------
# 出力（CSV / markdown / 図）
# ---------------------------------------------------------------------------
def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    import csv

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_phenomenon_outputs(results: dict[str, dict], out_dir: Path) -> None:
    """条件横断の per-phenomenon 表（CSV + markdown）。results: {label: analyze_full_main(...)}。"""
    labels = [lbl for lbl, r in results.items() if r.get("available")]
    if not labels:
        print("[analyze] full-main blimp が無い条件のみ。per-phenomenon 表はスキップ。")
        return
    phenoms = sorted({p for lbl in labels for p in results[lbl]["by_phenomenon"]})

    # CSV
    header = ["phenomenon", "n_paradigms"] + labels + (["diff"] if len(labels) == 2 else [])
    rows = []
    for p in phenoms:
        n = next(results[lbl]["by_phenomenon"][p]["n_paradigms"] for lbl in labels if p in results[lbl]["by_phenomenon"])
        vals = [results[lbl]["by_phenomenon"].get(p, {}).get("accuracy") for lbl in labels]
        row = [p, n] + [f"{v:.2f}" if v is not None else "" for v in vals]
        if len(labels) == 2 and all(v is not None for v in vals):
            row.append(f"{vals[0] - vals[1]:+.2f}")
        elif len(labels) == 2:
            row.append("")
        rows.append(row)
    # overall 行
    overalls = [results[lbl]["overall"] for lbl in labels]
    orow = ["__overall__", len(phenoms)] + [f"{v:.2f}" for v in overalls]
    if len(labels) == 2:
        orow.append(f"{overalls[0] - overalls[1]:+.2f}")
    rows.append(orow)

    _write_csv(out_dir / "per_phenomenon.csv", header, rows)
    (out_dir / "per_phenomenon.md").write_text(
        "## BLiMP per-phenomenon (full-main, 非加重平均)\n\n" + _md_table(header, rows) + "\n"
    )
    print(f"[analyze] per-phenomenon -> {out_dir/'per_phenomenon.csv'} / .md")


def write_trajectory_outputs(trajectories: dict[str, list[dict]], out_dir: Path) -> None:
    """条件横断の軌道 tidy CSV。trajectories: {label: rows}。"""
    any_rows = False
    header = ["condition", "revision", "words", "phenomenon", "accuracy", "n_paradigms"]
    rows = []
    for label, traj in trajectories.items():
        for r in traj:
            any_rows = True
            rows.append([label, r["revision"], r["words"], r["phenomenon"], f"{r['accuracy']:.4f}", r["n_paradigms"]])
    if not any_rows:
        print("[analyze] fast 軌道が無い。軌道 CSV はスキップ。")
        return
    _write_csv(out_dir / "trajectory.csv", header, rows)
    print(f"[analyze] trajectory -> {out_dir/'trajectory.csv'}")


def plot_figures(
    results: dict[str, dict], trajectories: dict[str, list[dict]], out_dir: Path
) -> None:
    """現象別棒グラフ・軌道ライン・ヒートマップを PNG 出力。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[analyze] matplotlib 不在のため作図スキップ: {e}")
        return

    labels = [lbl for lbl, r in results.items() if r.get("available")]

    # 図1: 現象別 accuracy 棒グラフ（条件横並び）
    if labels:
        phenoms = sorted({p for lbl in labels for p in results[lbl]["by_phenomenon"]})
        x = range(len(phenoms))
        width = 0.8 / max(len(labels), 1)
        fig, ax = plt.subplots(figsize=(12, 5))
        for j, lbl in enumerate(labels):
            ys = [results[lbl]["by_phenomenon"].get(p, {}).get("accuracy", float("nan")) for p in phenoms]
            ax.bar([i + j * width for i in x], ys, width=width, label=lbl)
        ax.set_xticks([i + width * (len(labels) - 1) / 2 for i in x])
        ax.set_xticklabels(phenoms, rotation=45, ha="right")
        ax.set_ylabel("accuracy (%)")
        ax.set_title("BLiMP per-phenomenon (full-main)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "phenomenon_bar.png", dpi=150)
        plt.close(fig)
        print(f"[analyze] figure -> {out_dir/'phenomenon_bar.png'}")

    # 図2: words_seen 軌道（現象別ライン、条件ごとに別ファイル）
    for label, traj in trajectories.items():
        if not traj:
            continue
        by_phenom: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for r in traj:
            if r["words"] is None:
                continue
            by_phenom[r["phenomenon"]].append((r["words"], r["accuracy"]))
        fig, ax = plt.subplots(figsize=(10, 6))
        for phenom, pts in sorted(by_phenom.items()):
            pts.sort()
            xs = [w for w, _ in pts]
            ys = [a for _, a in pts]
            style = dict(lw=2.5, color="black") if phenom == "__overall__" else dict(lw=1.2)
            ax.plot(xs, ys, marker="o", ms=3, label=phenom, **style)
        ax.set_xscale("log")
        ax.set_xlabel("words_seen (log)")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"BLiMP per-phenomenon trajectory — {label}")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"trajectory_{label}.png", dpi=150)
        plt.close(fig)
        print(f"[analyze] figure -> {out_dir/('trajectory_'+label+'.png')}")


# ---------------------------------------------------------------------------
def _parse_collate_args(items: list[str]) -> dict[str, Path]:
    """--collate LABEL=PATH（複数可）。'=' 無しなら stem をラベルにする。"""
    out: dict[str, Path] = {}
    for item in items:
        if "=" in item:
            label, _, path = item.partition("=")
        else:
            path = item
            label = Path(item).parent.name or Path(item).stem
        out[label] = Path(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collate", action="append", required=True,
        help="LABEL=path/to/all_full_preds_and_fast_scores_<backend>.json（複数指定可）",
    )
    parser.add_argument("--blimp-data", type=Path, default=DEFAULT_BLIMP_DATA,
                        help="BLiMP full_eval データ dir（現象マップ＆採点用）")
    parser.add_argument("--out", type=Path, default=_REPO / "analysis" / "outputs",
                        help="出力ディレクトリ")
    parser.add_argument("--no-plots", action="store_true", help="作図をスキップ")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    mapping = load_phenomenon_map(args.blimp_data)
    n_phenom = len({m["linguistics_term"] for m in mapping.values()})
    print(f"[analyze] 現象カテゴリ: {n_phenom} (linguistics_term), パラダイム: {len(mapping)}")

    collates = _parse_collate_args(args.collate)
    results: dict[str, dict] = {}
    trajectories: dict[str, list[dict]] = {}
    for label, path in collates.items():
        collate = json.loads(path.read_text())
        full = analyze_full_main(collate, mapping, args.blimp_data)
        results[label] = full
        trajectories[label] = analyze_trajectory(collate, mapping)
        if full.get("available"):
            print(f"[analyze] {label}: full-main BLiMP overall = {full['overall']:.2f} "
                  f"({len(full['per_paradigm'])} paradigms)")
        else:
            print(f"[analyze] {label}: full-main blimp 無し（fast 軌道のみ）")

    write_phenomenon_outputs(results, args.out)
    write_trajectory_outputs(trajectories, args.out)
    if not args.no_plots:
        plot_figures(results, trajectories, args.out)
    print(f"[analyze] 完了 -> {args.out}")


if __name__ == "__main__":
    main()
