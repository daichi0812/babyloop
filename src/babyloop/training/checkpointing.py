"""checkpoint管理。"""

from __future__ import annotations

from pathlib import Path


def milestone_label(words: int) -> str:
    """累積語数 → 公式命名ラベル。

    公式 evaluation-pipeline（collate_preds.py の ``chck_*M``）は **M単位**を
    仮定する（1000M を 1B とは書かない）ので、1M の倍数は常に M 単位で返す。
    1000000 -> '1M', 100000000 -> '100M', 1000000000 -> '1000M'。
    疎通用の 1M 未満は K 単位（提出対象外）。
    """
    if words % 1_000_000 == 0:
        return f"{words // 1_000_000}M"
    if words % 1_000 == 0:
        return f"{words // 1_000}K"
    return str(words)


def official_strict_milestones() -> list[int]:
    """BabyLM 2026 Strict の提出必須28点（累積語数, total not unique）。

    collate_preds.py の ``OTHER_FAST_REVISIONS`` と一致:
    1M刻み〜9M / 10M刻み〜90M / 100M刻み〜1000M。
    """
    return (
        [i * 1_000_000 for i in range(1, 10)]
        + [i * 10_000_000 for i in range(1, 10)]
        + [i * 100_000_000 for i in range(1, 11)]
    )


class CheckpointManager:
    """累積word数のマイルストーンでcheckpointを保存する。

    仮説(2)「劣化は訓練ダイナミクスで観測できる」の検証には中間
    checkpointが必須のため、step数ではなくword数を基準にする
    （configs/train/default.yaml の ``checkpoint_milestones``）。

    保存は HF 互換（``save_pretrained``）。公式 evaluation-pipeline が読めるよう
    tokenizer も同梱し、**``chck_<label>M`` 命名**（collate_preds.py がこの文字列を
    revision として直接仮定）に合わせる。

    Args:
        milestones: checkpointを保存する累積word数のリスト。
        output_dir: 保存先（Hydraのrun dir配下）。
        tokenizer: 同梱保存するtokenizer（任意）。
    """

    def __init__(self, milestones: list[int], output_dir: str, tokenizer=None):
        self.milestones = sorted(int(m) for m in milestones)
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self._saved: set[int] = set()

    def maybe_save(self, model, words_seen: int, step: int) -> list[str]:
        """未保存のマイルストーンを超えていたらcheckpointを保存する。

        Returns: 今回保存した checkpoint パスのリスト。
        """
        saved_now = []
        for m in self.milestones:
            if m in self._saved or words_seen < m:
                continue
            path = self.output_dir / f"chck_{milestone_label(m)}"
            self._save(model, path)
            self._saved.add(m)
            saved_now.append(str(path))
        return saved_now

    def save_final(self, model, words_seen: int) -> str:
        """学習終了時の最終checkpointを保存する（提出時の HF ``main`` 相当）。"""
        path = self.output_dir / "ckpt_final"
        self._save(model, path)
        return str(path)

    def _save(self, model, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
