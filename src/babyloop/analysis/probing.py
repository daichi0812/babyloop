"""中間表現のprobing。

仮説(2)（言語表現への破壊的干渉）と仮説(4)（groundingの利得は
concreteな概念で大きい）を、概念カテゴリ別のprobingで検証する。
"""


class Prober:
    """checkpointの中間表現（各ループ・各層）に対してprobing classifierを学習する。

    概念カテゴリは文法（grammatical）/ grounded / 物理（physical）で分離し、
    カテゴリ×checkpoint（word数マイルストーン）×ループ回の格子で
    probing精度を測る。

    Args:
        checkpoint_path: 対象checkpoint。
        categories: probing対象の概念カテゴリ。
    """

    def __init__(self, checkpoint_path: str, categories: list[str] | None = None):
        self.checkpoint_path = checkpoint_path
        self.categories = categories or ["grammatical", "grounded", "physical"]

    def run(self):
        """probingを実行し、カテゴリ→精度のdictを返す。"""
        raise NotImplementedError
