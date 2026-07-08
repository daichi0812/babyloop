"""BPEトークナイザの学習・読み込み。

前処理（TextPreprocessor）と checkpoint 同梱の両方で使う。公式
evaluation-pipeline は ``AutoProcessor.from_pretrained(..., trust_remote_code=True)``
→ fast tokenizer 前提（``return_offsets_mapping`` を使う）でトークナイザを読むため、
``PreTrainedTokenizerFast`` 形式（tokenizer.json 同梱）で保存する。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def train_bpe_tokenizer(
    text_iterator: Iterable[str],
    vocab_size: int,
    save_dir: str | None = None,
    max_seq_len: int = 512,
) -> PreTrainedTokenizerFast:
    """ByteLevel BPE を学習し ``PreTrainedTokenizerFast`` を返す（任意で保存）。"""
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)

    # 各系列を BOS ... EOS で囲む（因果LMの文境界）。
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, bos_id), (EOS_TOKEN, eos_id)],
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        model_max_length=max_seq_len,
    )
    if save_dir is not None:
        fast.save_pretrained(save_dir)
    return fast


def load_tokenizer(path: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(path)


def iter_texts_from_lines(paths: Iterable[str]) -> Iterator[str]:
    """テキストファイル群を1行=1サンプルでyieldする。"""
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
