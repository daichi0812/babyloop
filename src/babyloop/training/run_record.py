"""run ごとの軽量レコード（Git追跡）。

Mac(開発)↔A100(学習) を行き来するため、**重い成果物（checkpoint・tokens・
生データ）はGit外**に置きつつ、再現に要る軽量物だけを Git で同期する:

  runs/<name>/seed_<S>/
    ├── run.json      … config スナップショット・seed・git sha・device・所要時間・最終メトリクス
    ├── metrics.jsonl … step 毎の loss/lr/words_seen（+ ts/elapsed_s）
    ├── train.log     … 生ログ（各行に時刻つき）
    └── eval.json     … 評価スコア（scripts/evaluate.py が追記）

`runs/` はトップレベル（機械生成の成果物置き場）。`docs/experiments/` はキュレーション
された index（人手運用）。役割を分ける（[docs-notion-split]）。
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_ROOT = Path("runs")

_SEED_DIR_RE = re.compile(r"seed_(\d+)")


def run_id_from_checkpoint(checkpoint: str | Path) -> tuple[str, int] | None:
    """checkpoint パス `outputs/<name>/seed_<S>/...` から `(name, seed)` を取り出す。

    eval 記録先（`runs/<name>/seed_<S>/eval.json`）は **評価した checkpoint** に一致させたい。
    記録先を `cfg.train.seed` で決めると、`+checkpoint=...seed_43...` を渡しても
    `train.seed` を上書きし忘れると default(=42) の記録を**サイレント上書き**する footgun に
    なるため、checkpoint のパスを single source of truth にする。

    `seed_<N>` セグメントとその直前の name を返す。一致しなければ `None`（呼び出し側で
    cfg にフォールバック）。絶対/相対どちらのパスでも、また末尾が `ckpt_final` でも
    `chck_500M` でも動く。
    """
    parts = Path(checkpoint).parts
    for i, seg in enumerate(parts):
        m = _SEED_DIR_RE.fullmatch(seg)
        if m and i >= 1:
            return parts[i - 1], int(m.group(1))
    return None


def git_commit() -> dict:
    """現在の commit と dirty 状態（best-effort）。"""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return {"commit": sha, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def run_dir(name: str, seed: int, root: Path | str = RUN_ROOT) -> Path:
    return Path(root) / name / f"seed_{seed}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunRecorder:
    """学習1 run の軽量ログを Git追跡ディレクトリへ書き出す。"""

    def __init__(self, name: str, seed: int, root: Path | str = RUN_ROOT, resume: bool = False):
        self.dir = run_dir(name, seed, root)
        self.dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "w"  # resume 時は過去ログ/メトリクスへ追記
        self._log = (self.dir / "train.log").open(mode, encoding="utf-8")
        self._metrics = (self.dir / "metrics.jsonl").open(mode, encoding="utf-8")
        self.name = name
        self.seed = seed
        self._t0 = time.monotonic()
        self._started = _now()

    def log(self, msg: str) -> None:
        """stdout と train.log の両方へ出力（各行に UTC 時刻つき）。"""
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def metric(self, **fields) -> None:
        """step 毎メトリクスを metrics.jsonl へ1行追記（ts/elapsed_s を自動付与）。"""
        record = {"ts": _now(), "elapsed_s": round(time.monotonic() - self._t0, 2), **fields}
        self._metrics.write(json.dumps(record) + "\n")
        self._metrics.flush()

    def finalize(self, config: dict, summary: dict, device: str) -> Path:
        """run.json を書き出して run を締める（所要時間・throughput つき）。"""
        duration = round(time.monotonic() - self._t0, 2)
        throughput = {}
        if duration > 0:
            if "steps" in summary:
                throughput["steps_per_sec"] = round(summary["steps"] / duration, 3)
            if "words_seen" in summary:
                throughput["words_per_sec"] = round(summary["words_seen"] / duration, 1)
        record = {
            "name": self.name,
            "seed": self.seed,
            "started_at": self._started,
            "finished_at": _now(),
            "duration_s": duration,
            "throughput": throughput,
            "device": device,
            "git": git_commit(),
            "config": config,
            "summary": summary,
        }
        path = self.dir / "run.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        self._log.close()
        self._metrics.close()
        return path


def write_eval_record(name: str, seed: int, payload: dict, root: Path | str = RUN_ROOT) -> Path:
    """評価スコアを run ディレクトリの eval.json へ書き出す（scripts/evaluate.py 用）。"""
    d = run_dir(name, seed, root)
    d.mkdir(parents=True, exist_ok=True)
    payload = {"recorded_at": _now(), "git": git_commit(), **payload}
    path = d / "eval.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path
