"""BabyLM 2026 leaderboard へ collate 出力を提出する（gradio_client API `/submit_and_refresh`）。

公式 Space (`BabyLM-community/BabyLM-Leaderboard-2026`) の Submit ボタンが叩くのと同じ関数を
`gradio_client` で呼ぶ。**UI は使わない**：結果 JSON は ~62MB でブラウザのファイル添付上限を超え、
agent による代行提出も遮断されるため。**最終実行はユーザー本人**が行うこと（公開 board への提出）。

前提: HF write token（`~/.cache/huggingface/token` か環境変数 `HF_TOKEN`）。
使い方:
    uv run --with gradio_client python scripts/submit_leaderboard.py <cfg.json>

cfg.json のキー（必須: model_name / hf_repo / results_file / params / flops / gpu_train_hours /
description、他は既定値あり）:
    model_name       board 表示名（既存と同名なら上書き更新。例 "babyloop-loop-text (causal)"）
    hf_repo          実在 repo を入れると board で青字リンク化（例 "daichi812/babyloop-loop-text"）
    results_file     collate 出力 all_full_preds_and_fast_scores_causal.json のパス
    contributions    list[str]（例 ["Controlled experiments", "Architectural innovations"]）
    seed, params, flops, gpu_train_hours, description
    （既定: track=strict / model_type=Decoder only / base_arch=Llama / lr_scheduler=cosine /
      epochs=10 / tokenizer=BPE 16k / num_heads=12 / max_seq_len=512 / gpu_dev_hours=100 /
      training_data=BabyLM strict / max_lr=0.0015 / optimizer=AdamW / batch_tokens=262144 /
      vocab=16000 / num_layers=12）

メモ: gradio_client 2.5.0 は `Client(src, token=...)`（kwarg は `hf_token` ではない）。strict では
多言語 predictions(param_5) と other-hyperparameters file(param_31) は None で通る。欠損タスクは
0 点で valid（GLUE 必須でない）。2026 は multimodal/interaction トラック廃止＝Strict 統合のため
image-text セルも track="strict"。
"""

from __future__ import annotations

import json
import os
import sys

SPACE = "BabyLM-community/BabyLM-Leaderboard-2026"


def _token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    path = os.path.expanduser("~/.cache/huggingface/token")
    return open(path).read().strip() if os.path.exists(path) else None


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: submit_leaderboard.py <cfg.json>")
    cfg = json.load(open(sys.argv[1]))
    g = cfg.get

    from gradio_client import Client, handle_file

    client = Client(SPACE, token=_token())
    print(f"submitting: {g('model_name')}  track={g('track', 'strict')}  repo={g('hf_repo')}")
    res = client.predict(
        g("model_name"),                       # 0  model name（同名で上書き更新）
        g("revision", "main"),                 # 1  revision
        g("hf_repo"),                          # 2  HF repo（実在で青字リンク）
        g("track", "strict"),                  # 3  track
        handle_file(g("results_file")),        # 4  results JSON（collate 出力）
        None,                                  # 5  多言語 predictions（strict は None）
        g("model_type", "Decoder only"),       # 6
        g("contributions", []),                # 7  list[str]
        g("base_arch", "Llama"),               # 8
        g("lr_scheduler", "cosine"),           # 9
        g("epochs", 10),                       # 10
        g("tokenizer", "BPE 16k (BabyLM strict)"),  # 11
        str(g("seed", 42)),                    # 12
        g("num_heads", 12),                    # 13
        g("max_seq_len", 512),                 # 14
        g("gpu_dev_hours", 100),               # 15  開発総 GPU 時間（概算）
        g("training_data", "BabyLM strict"),   # 16
        0,                                     # 17  custom dataset 語数（公式なら 0）
        "",                                    # 18  custom dataset ジャンル
        g("max_lr", 0.0015),                   # 19
        g("optimizer", "AdamW"),               # 20
        g("batch_tokens", 262144),             # 21  平均バッチ（tokens）
        g("vocab", 16000),                     # 22  token set size
        g("num_layers", 12),                   # 23
        g("params"),                           # 24  総パラメータ数
        g("flops"),                            # 25  概算 FLOPS
        g("gpu_train_hours"),                  # 26  このモデルの学習 GPU 時間
        "Not applicable",                      # 27  human annotation
        "",                                    # 28  custom preprocessing
        "Not applicable",                      # 29  synthetic/augmentation
        g("description"),                      # 30  説明（空不可）
        None,                                  # 31  other hyperparameters file
        "",                                    # 32  teacher models
        False, False, False,                   # 33-35  en/nl/zh（multilingual 用）
        0, 0, 0,                               # 36-38  tokens per lang
        api_name="/submit_and_refresh",
    )
    md = res[0] if isinstance(res, (list, tuple)) else res
    print("=== RESULT ===")
    print(md)


if __name__ == "__main__":
    main()
