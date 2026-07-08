"""Word budget計上。

BabyLMの固定word budget（100M words）の遵守はこのモジュールが単一の
責任を持つ。マルチモーダルtrackではキャプションの語数も計上する。
計上ロジックの正しさは tests/test_word_budget.py で検証する
（壊れると提出が無効になるため）。
"""


class BudgetExceededError(RuntimeError):
    """word budgetを超過した場合に送出される。"""


class WordBudgetTracker:
    """学習データの累積語数を追跡し、budget超過を検出する。

    Args:
        budget: 許容される総語数（configs/data/*.yaml の ``word_budget``）。
    """

    def __init__(self, budget: int):
        self.budget = budget
        self.consumed = 0

    def add(self, n_words: int) -> None:
        """語数を計上する。超過したら BudgetExceededError を送出する。"""
        if n_words < 0:
            raise ValueError("n_words must be non-negative")
        self.consumed += n_words
        if self.consumed > self.budget:
            raise BudgetExceededError(
                f"word budget exceeded: {self.consumed} > {self.budget}"
            )

    @property
    def remaining(self) -> int:
        """残り語数。"""
        return self.budget - self.consumed

    @staticmethod
    def count_words(text: str) -> int:
        """BabyLM公式ルールに従った語数カウント（空白区切り）。

        BabyLM のword budgetは空白トークン単位で定義される。テキスト・
        キャプションの双方に同一規則を適用する。
        """
        return len(text.split())
