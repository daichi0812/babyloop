"""EvaluationPipelineWrapper の report パースのテスト。

背景: 以前 `_parse_avg_accuracy` は report 欠損・破損で黙って ``None`` を返し、
subprocess が成功しているのにスコア欠落へ気づけなかった。fail-loud（例外）に変えた
回帰を守る。
"""

import subprocess
from pathlib import Path

import pytest

from babyloop.evaluation import pipeline_wrapper as pw
from babyloop.evaluation.pipeline_wrapper import EvaluationPipelineWrapper

_REPORT = """\
TEMPERATURE: 1.00

### UID ACCURACY
anaphor_agreement: 99.00

### AVERAGE ACCURACY
75.22
"""


def _wrapper() -> EvaluationPipelineWrapper:
    # __init__ は IO せずパス整形のみなのでダミー checkpoint で良い。
    return EvaluationPipelineWrapper("outputs/std_text/seed_42/ckpt_final")


def test_parse_avg_accuracy_reads_value(tmp_path):
    report = tmp_path / "best_temperature_report.txt"
    report.write_text(_REPORT)
    assert _wrapper()._parse_avg_accuracy(report) == 75.22


def test_parse_avg_accuracy_raises_when_report_missing(tmp_path):
    with pytest.raises(RuntimeError, match="report が見つかりません"):
        _wrapper()._parse_avg_accuracy(tmp_path / "absent.txt")


def test_parse_avg_accuracy_raises_when_no_average_line(tmp_path):
    report = tmp_path / "best_temperature_report.txt"
    report.write_text("TEMPERATURE: 1.00\n\n### UID ACCURACY\nfoo: 50.00\n")
    with pytest.raises(RuntimeError, match="AVERAGE ACCURACY 行がありません"):
        _wrapper()._parse_avg_accuracy(report)


def test_parse_avg_accuracy_raises_when_value_unparseable(tmp_path):
    report = tmp_path / "best_temperature_report.txt"
    report.write_text("### AVERAGE ACCURACY\nNaN-ish???\n")
    with pytest.raises(RuntimeError, match="パースできません"):
        _wrapper()._parse_avg_accuracy(report)


def test_run_reads_report_from_given_output_dir(tmp_path, monkeypatch):
    """名前空間化の契約: run() は渡した output_dir 配下の report を読む。

    背景: model stem は全セル共通の `ckpt_final` なので、固定 output_dir だと
    全セル×seed が衝突する。caller が output_dir を分離すれば read もそこに追従する
    ことを、subprocess（公式 pipeline）を mock して回帰固定する。
    """
    ckpt = tmp_path / "outputs" / "loop_mm" / "seed_42" / "ckpt_final"
    ckpt.mkdir(parents=True)
    out_dir = tmp_path / "eval_results" / "loop_mm" / "seed_42"
    wrapper = EvaluationPipelineWrapper(str(ckpt))
    # eval データ取得は no-op（submodule データ DL を避ける）。
    monkeypatch.setattr(wrapper, "_ensure_eval_data", lambda eval_dir: None)

    # 公式 pipeline の出力レイアウト（run.py:153）を再現して report を置く mock。
    def fake_run(cmd, cwd=None, check=False):
        od = Path(cmd[cmd.index("--output_dir") + 1])
        task = cmd[cmd.index("--task") + 1]
        dataset = Path(cmd[cmd.index("--data_path") + 1]).stem
        report_dir = od / "ckpt_final" / "main" / "zero_shot" / "causal" / task / dataset
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "best_temperature_report.txt").write_text(_REPORT)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(pw.subprocess, "run", fake_run)

    scores = wrapper.run(["blimp"], split="full", output_dir=str(out_dir))
    assert scores == {"blimp": 75.22}
    expected = (
        out_dir / "ckpt_final" / "main" / "zero_shot" / "causal"
        / "blimp" / "blimp_filtered" / "best_temperature_report.txt"
    )
    assert expected.exists()
