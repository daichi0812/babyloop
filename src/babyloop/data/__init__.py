"""前処理・word budget計上・dataloader。

テキスト専用とマルチモーダルは共通インターフェース
（BasePreprocessor / BaseDataModule）を実装する。
"""

from babyloop.data.word_budget import WordBudgetTracker

__all__ = ["WordBudgetTracker"]
