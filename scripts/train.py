"""学習エントリポイント。

使い方:
    uv run python scripts/train.py experiment=std_text
    uv run python scripts/train.py experiment=std_text_smoke
    uv run python scripts/train.py experiment=std_text train.seed=43
    # 中断からの再開（optimizer/scheduler/step/words/RNG を復元）:
    uv run python scripts/train.py experiment=std_text +resume=outputs/std_text/seed_42
"""

import hydra
from omegaconf import DictConfig

from babyloop.training.trainer import Trainer


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
