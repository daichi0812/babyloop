"""BabyLM公式evaluation-pipelineのラッパー。

pipeline本体は ``third_party/evaluation-pipeline`` submodule にコミット固定で
導入する（[ADR-0004]・docs/reproduce.md）。評価器のバージョン差は
2×2セル間の比較を汚染するため、全セルを同一revで評価すること。

実体は公式 `evaluation_pipeline.sentence_zero_shot.run` を **同一Python環境**の
subprocess として起動する（モデルは ``AutoModelForCausalLM.from_pretrained(
checkpoint, trust_remote_code=True)`` で読まれるため、checkpoint に
modeling/config ファイルとtokenizerが同梱されている必要がある）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# このファイル: src/babyloop/evaluation/pipeline_wrapper.py → repo root は parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STRICT_DIR = _REPO_ROOT / "third_party" / "evaluation-pipeline" / "strict"

# fast（中間ckpt用・subsample）/ full（最終モデル用）それぞれのタスク→データセット名。
_FAST_DATASETS = {
    "blimp": "blimp_fast",
    "supplement": "supplement_fast",
    "ewok": "ewok_fast",
    "entity_tracking": "entity_tracking_fast",
}
_FULL_DATASETS = {
    "blimp": "blimp_filtered",
    "supplement": "supplement_filtered",
    "ewok": "ewok_filtered",
    "entity_tracking": "entity_tracking",
    "comps": "comps",
}


class EvaluationPipelineWrapper:
    """checkpointを公式pipelineのzero-shotタスクで評価する。

    Args:
        checkpoint_path: 評価対象のcheckpoint（save_pretrained 済みディレクトリ）。
        pipeline_rev: 使用するevaluation-pipelineのコミットrev（記録用）。
        strict_dir: 公式pipelineの strict ディレクトリ（既定: submodule）。
        backend: 評価バックエンド（causal LM は ``"causal"``）。
    """

    def __init__(
        self,
        checkpoint_path: str,
        pipeline_rev: str | None = None,
        strict_dir: str | Path | None = None,
        backend: str = "causal",
    ):
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.pipeline_rev = pipeline_rev
        self.strict_dir = Path(strict_dir) if strict_dir else _DEFAULT_STRICT_DIR
        self.backend = backend

    def _ensure_eval_data(self, eval_dir: Path) -> None:
        if eval_dir.exists():
            return
        # 公式の download スクリプトで evaluation_data を取得（公開データセット）。
        subprocess.run(
            [sys.executable, "-m", "scripts.download_evals"],
            cwd=self.strict_dir,
            check=True,
        )

    def _parse_avg_accuracy(self, report_path: Path) -> float:
        """report の AVERAGE ACCURACY を読む。

        欠損・破損を黙って ``None`` に化けさせず例外で顕在化させる（subprocess は
        ``check=True`` で成功しているのに report が無い／壊れている＝想定外で、
        None を記録するとスコア欠落に気づけないため）。
        """
        if not report_path.exists():
            raise RuntimeError(
                f"評価 report が見つかりません: {report_path}"
                "（subprocess は成功したが出力が無い）"
            )
        lines = report_path.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "### AVERAGE ACCURACY" and i + 1 < len(lines):
                try:
                    return float(lines[i + 1].strip())
                except ValueError as e:
                    raise RuntimeError(
                        f"AVERAGE ACCURACY をパースできません: {report_path} の "
                        f"'{lines[i + 1].strip()}'"
                    ) from e
        raise RuntimeError(f"report に AVERAGE ACCURACY 行がありません: {report_path}")

    def run(
        self,
        tasks: list[str],
        split: str = "fast",
        output_dir: str | Path = "eval_results",
    ) -> dict[str, float]:
        """指定タスクを実行し、タスク名→平均accuracy(%) のdictを返す。

        Note:
            公式 pipeline は出力を ``<output_dir>/<checkpoint stem>/...`` に書き、
            stem は全 2×2 セルで共通の ``ckpt_final`` になりがち。**呼び出し側で
            ``output_dir`` を checkpoint（name/seed）ごとに名前空間化すること**。
            既定の ``"eval_results"`` のまま複数セルを評価すると同一パスへ書き込み、
            並行評価で別 run の report を読む取り違えが起きうる（``scripts/evaluate.py``
            参照）。
        """
        datasets = _FAST_DATASETS if split == "fast" else _FULL_DATASETS
        eval_dir = self.strict_dir / "evaluation_data" / (
            "fast_eval" if split == "fast" else "full_eval"
        )
        self._ensure_eval_data(eval_dir)

        output_dir = Path(output_dir).resolve()
        model_name = Path(self.checkpoint_path).stem
        scores: dict[str, float] = {}

        for task in tasks:
            if task not in datasets:
                raise ValueError(f"未対応タスク: {task}（対応: {list(datasets)}）")
            dataset = datasets[task]
            data_path = eval_dir / dataset
            cmd = [
                sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
                "--model_path_or_name", self.checkpoint_path,
                "--backend", self.backend,
                "--task", task,
                "--data_path", str(data_path),
                "--output_dir", str(output_dir),
                "--save_predictions",
            ]
            subprocess.run(cmd, cwd=self.strict_dir, check=True)

            report = (
                output_dir / model_name / "main" / "zero_shot" / self.backend
                / task / dataset / "best_temperature_report.txt"
            )
            scores[task] = self._parse_avg_accuracy(report)

        return scores
