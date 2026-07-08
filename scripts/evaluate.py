"""評価エントリポイント（公式evaluation-pipelineのラッパー経由）。

使い方:
    uv run python scripts/evaluate.py experiment=std_text_smoke \
        +checkpoint=outputs/std_text_smoke/seed_42/ckpt_final
    uv run python scripts/evaluate.py experiment=std_text +checkpoint=<path> +eval_split=full
"""

import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig

from babyloop.evaluation.pipeline_wrapper import EvaluationPipelineWrapper
from babyloop.training.run_record import run_id_from_checkpoint, write_eval_record

_SUBMODULE = Path(__file__).resolve().parent.parent / "third_party" / "evaluation-pipeline"


def _pipeline_rev() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(_SUBMODULE), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if checkpoint is None:
        raise SystemExit("評価には +checkpoint=<path> が必要です。")
    split = cfg.get("eval_split", "fast")
    tasks = list(cfg.get("eval_tasks", ["blimp"]))

    # 記録先は「評価した checkpoint のパス」を真実とする（cfg.train.seed 依存だと
    # +checkpoint だけ変えて seed を渡し忘れたとき別 seed を上書きする footgun になる）。
    cfg_name, cfg_seed = cfg.name, int(cfg.train.seed)
    derived = run_id_from_checkpoint(checkpoint)
    if derived is not None:
        name, seed = derived
        if name != cfg_name or seed != cfg_seed:
            print(
                f"[evaluate] WARNING: 記録先は checkpoint パス由来の {name}/seed_{seed} を使います"
                f"（cfg は {cfg_name}/seed_{cfg_seed}）。+checkpoint と train.seed の不一致を検出。"
            )
    else:
        name, seed = cfg_name, cfg_seed
        print(
            f"[evaluate] WARNING: checkpoint パスから seed を特定できません "
            f"（`outputs/<name>/seed_<S>/...` 形式でない）。記録先は cfg 由来の {name}/seed_{seed}。"
        )

    rev = _pipeline_rev()
    wrapper = EvaluationPipelineWrapper(checkpoint, pipeline_rev=rev)
    # 出力先を name/seed で名前空間化する。公式 pipeline は出力 dir を
    # `<output_dir>/<checkpoint stem>/...` に組み立て、stem は全セル共通の `ckpt_final`
    # なので、固定の "eval_results" のままだと全セル×seed が同一パスへ書き、
    # (1) on-disk 結果が判別不能・最後の1件で毎回上書き (2) 並行評価時に別 run の
    # report を読んで誤記録、という footgun になる。name/seed で分離して根絶する。
    output_dir = f"eval_results/{name}/seed_{seed}"
    scores = wrapper.run(tasks, split=split, output_dir=output_dir)
    print(f"[evaluate] checkpoint={checkpoint} -> {name}/seed_{seed} pipeline_rev={rev} "
          f"split={split} output_dir={output_dir}")
    print(f"[evaluate] scores={scores}")

    record = write_eval_record(
        name,
        seed,
        {
            "checkpoint": str(checkpoint),
            "name": name,    # 記録を自己記述に（dir 取り違えを後から検知できる）
            "seed": seed,
            "pipeline_rev": rev,
            "split": split,
            "tasks": tasks,
            "scores": scores,
        },
    )
    print(f"[evaluate] eval record -> {record}")


if __name__ == "__main__":
    main()
