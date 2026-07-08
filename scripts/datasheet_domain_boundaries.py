#!/usr/bin/env python3
"""text prefix ドメイン境界の確定（分析のみ・学習なし）。

Task A: 公式 HF repo の6ドメインファイルを取得し、train.txt の結合順を
        greedy 全行照合（strip 後の逐行一致）で確定する。
Task B: 実装と同一規則（strip→空行skip→len(split)、超過行は含めず停止）で
        50M 語境界のドメイン位置と prefix/全量のドメイン内訳を出す。
Task C: captions 先頭 3,037,190 行の ln_/cc_ 集計と caption_words.npy の突き合わせ。
出力: /data/hotta/babyloop/data/text/domain_boundaries.json
"""
import json, sys, time, itertools
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import numpy as np

REPO = "BabyLM-community/BabyLM-2026-Strict"
DATA = Path("/data/hotta/babyloop/data")
TRAIN = DATA / "text/train.txt"
DOMAINS = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
TEXT_BUDGET = 50_000_000          # round(1e8 * 0.5)
EXPECT_PREFIX = 49_999_988
EXPECT_LINES = 11_601_896
EXPECT_TOTAL_WORDS = 100_000_000
PROBE = 1000

log = lambda *a: (print(*a), sys.stdout.flush())

# --- Task A-1: ファイル取得 ---
api = HfApi()
info = api.repo_info(REPO, repo_type="dataset")
revision = info.sha
files = api.list_repo_files(REPO, repo_type="dataset")
chosen = {}
for d in DOMAINS:
    cand = [f for f in files if d in f]
    if len(cand) > 1:
        pref = [f for f in cand if "train" in f.lower()]
        cand = pref or cand
    assert len(cand) == 1, f"{d}: 候補が一意でない {cand}"
    chosen[d] = cand[0]
log("[files]", json.dumps(chosen, ensure_ascii=False))
paths = {d: Path(hf_hub_download(REPO, f, repo_type="dataset")) for d, f in chosen.items()}
sizes = {d: paths[d].stat().st_size for d in DOMAINS}
log("[bytes]", sizes, "sum:", sum(sizes.values()))

def nonempty(p):
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                yield s

# --- Task A-2: 各ドメインの非空行数・語数 ---
dom_lines, dom_words = {}, {}
for d in DOMAINS:
    n = w = 0
    for s in nonempty(paths[d]):
        n += 1
        w += len(s.split())
    dom_lines[d], dom_words[d] = n, w
    log(f"[domain] {d}: lines={n:,} words={w:,}")
log("[check] lines合計:", sum(dom_lines.values()), "期待:", EXPECT_LINES)
log("[check] words合計:", sum(dom_words.values()), "期待:", EXPECT_TOTAL_WORDS)

# --- Task A-3 + B: greedy 全行照合で結合順を確定しつつ境界を測る ---
train = nonempty(TRAIN)
remaining = list(DOMAINS)
order = []
cum_words = 0
train_line_no = 0            # 消費済み train 行数
prefix_total = None          # 停止規則での prefix 語数
boundary = None              # (domain, words_into_domain, line_into_domain)
per_domain_prefix = {d: 0 for d in DOMAINS}
tail_cut = None
mismatch = None

buf = list(itertools.islice(train, PROBE))
while remaining and buf:
    probe_n = min(PROBE, len(buf))
    match_d = None
    for d in remaining:
        head = list(itertools.islice(nonempty(paths[d]), probe_n))
        if head == buf[:probe_n]:
            match_d = d
            break
    if match_d is None:
        mismatch = {"at_train_line": train_line_no + 1, "train_head": buf[0][:200]}
        log("[FATAL] 先頭プローブがどのドメインとも不一致:", mismatch)
        break
    log(f"[order] 次のドメイン = {match_d}")
    order.append(match_d)
    remaining.remove(match_d)
    dwords = dlines = 0
    dom_iter = nonempty(paths[match_d])
    exhausted_train = False
    for dline in dom_iter:
        if buf:
            tline = buf.pop(0)
        else:
            tline = next(train, None)
            if tline is None:
                # train.txt がドメイン途中で尽きた（=100M 打ち切りの尻尾）
                tail_cut = {"domain": match_d, "lines_included": dlines,
                            "words_included": dwords,
                            "file_lines": dom_lines[match_d], "file_words": dom_words[match_d]}
                log("[tail] train.txt がドメイン途中で終了:", tail_cut)
                exhausted_train = True
                break
        if dline != tline:
            mismatch = {"domain": match_d, "at_train_line": train_line_no + 1,
                        "domain_line": dline[:200], "train_line": tline[:200]}
            log("[FATAL] 行不一致:", json.dumps(mismatch, ensure_ascii=False)[:500])
            break
        train_line_no += 1
        dlines += 1
        w = len(dline.split())
        # 停止規則（超過する行は含めない）で prefix を測る
        if prefix_total is None:
            if cum_words + w > TEXT_BUDGET:
                prefix_total = cum_words
                boundary = {"domain": match_d, "words_into_domain": dwords,
                            "lines_into_domain": dlines - 1}
                log(f"[boundary] 50M 境界: {match_d} 内 {dwords:,} 語地点 / prefix合計 {cum_words:,}")
            else:
                per_domain_prefix[match_d] += w
        cum_words += w
        dwords += w
    if mismatch or exhausted_train:
        break
    # 次ドメインのプローブ用バッファを補充
    buf = list(itertools.islice(train, PROBE))

leftover = sum(1 for _ in train) + len(buf)
log("[walk] 消費 train 行:", train_line_no, " 余り:", leftover, " 累計語:", cum_words)

# --- Task C: caption 検証 ---
cap_counts = {"ln": 0, "cc": 0, "other": 0}
with open(DATA / "mm/captions/captions.jsonl", encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i >= 3_037_190:
            break
        iid = json.loads(line)["image_id"]
        k = "ln" if iid.startswith("ln_") else ("cc" if iid.startswith("cc_") else "other")
        cap_counts[k] += 1
cw = np.load(DATA / "mm/processed/captions/caption_words.npy")
cap_npy = {"n": int(len(cw)), "sum_words": int(cw.sum())}
log("[captions] head3,037,190:", cap_counts, " npy:", cap_npy)

# --- 出力 ---
out = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "generator": "datasheet_domain_boundaries.py (see repo scripts/a100/ after review)",
    "hf_repo": REPO, "hf_revision": revision, "hf_files": chosen, "hf_bytes": sizes,
    "word_rule": "line.strip(); skip empty; len(line.split()); stop when cum+w > budget (line excluded)",
    "text_budget": TEXT_BUDGET,
    "concat_order": order,
    "order_verification": "greedy full-line match (stripped), fail-fast",
    "mismatch": mismatch, "tail_cut": tail_cut,
    "train_lines_walked": train_line_no, "train_lines_leftover": leftover,
    "domain_lines": dom_lines, "domain_words": dom_words,
    "checks": {
        "lines_sum": sum(dom_lines.values()), "lines_expected": EXPECT_LINES,
        "words_sum": sum(dom_words.values()), "words_expected": EXPECT_TOTAL_WORDS,
        "walk_total_words": cum_words,
        "prefix_total": prefix_total, "prefix_expected": EXPECT_PREFIX,
        "prefix_match": prefix_total == EXPECT_PREFIX,
    },
    "boundary_50M": boundary,
    "prefix_domain_words": per_domain_prefix,
    "prefix_domain_pct": {d: (per_domain_prefix[d] / prefix_total * 100 if prefix_total else None)
                          for d in DOMAINS},
    "full_domain_pct": {d: dom_words[d] / max(cum_words, 1) * 100 for d in DOMAINS},
    "caption_check": {"head_counts": cap_counts, "npy": cap_npy,
                      "expected": {"ln": 767736, "cc": 2269454, "sum_words": 49999996, "n": 3037190}},
}
dest = DATA / "text/domain_boundaries.json"
dest.write_text(json.dumps(out, ensure_ascii=False, indent=2))
log("[write]", dest)
log("DATASHEET_DONE" if not mismatch else "DATASHEET_FAILED")
