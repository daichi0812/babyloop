"""前処理エントリポイント。

使い方:
    uv run python scripts/preprocess.py data=text_only
    uv run python scripts/preprocess.py experiment=std_text_smoke
"""

import hydra
from omegaconf import DictConfig

from babyloop.data.preprocessing import MultimodalPreprocessor, TextPreprocessor


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    preprocessor = (
        MultimodalPreprocessor(cfg)
        if cfg.data.modality == "multimodal"
        else TextPreprocessor(cfg)
    )
    preprocessor.run()


if __name__ == "__main__":
    main()
