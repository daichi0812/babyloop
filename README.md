# babyloop

Code for a BabyLM Challenge 2026 (Strict track) submission: a controlled study of
whether **looped (recurrent-depth) transformers** mitigate the text-benchmark
degradation induced by **multimodal pretraining** under a fixed 100M-word
budget.

**Paper:** *Loop Before You Leap: What Recurrent Depth Can and Cannot Buy in
Data-Limited Pretraining* — under review at the BabyLM 2026 workshop
(arXiv preprint in preparation).

> Hypothesis: multimodal pretraining degrades text-only abilities (e.g.
> BLiMP) under a tight word budget. Looped computation adds effective depth —
> latent iterative processing — without extra data or parameters, and may
> mitigate that degradation.

## Experimental conditions

The conditions cross architecture (standard vs. looped) with modality
(text-only vs. multimodal); each condition is one file in `configs/experiment/`.
The paper analyzes these as three planned comparisons (compute allocation,
cross-modal integration, and attribution of the multimodal penalty) rather
than as a full factorial design:

|                      | text-only    | multimodal  |
| -------------------- | ------------ | ----------- |
| standard Transformer | `std_text`   | `std_mm`    |
| looped Transformer   | `loop_text`  | `loop_mm`   |

A single `LoopedTransformer` implementation covers both architectures: with
loop count `K=1` it reduces *exactly* to a standard Transformer
(`tests/test_k1_equivalence.py` pins this invariant).

## Released artifacts

- **Models** (Hugging Face, one repo per cell, final model on `main` + 28
  milestone checkpoints as `chck_*M` branches):
  [`babyloop-std-text`](https://huggingface.co/daichi812/babyloop-std-text) ·
  [`babyloop-std-mm`](https://huggingface.co/daichi812/babyloop-std-mm) ·
  [`babyloop-loop-text`](https://huggingface.co/daichi812/babyloop-loop-text) ·
  [`babyloop-loop-mm`](https://huggingface.co/daichi812/babyloop-loop-mm)
- **Multimodal training corpus** with a datasheet (assembled 50M/50M
  text+caption mixture and provenance manifests):
  [`daichi812/babyloop-mm-corpus`](https://huggingface.co/datasets/daichi812/babyloop-mm-corpus)
- **Leaderboard**: [BabyLM-Leaderboard-2026](https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026)

## Setup

Dependencies are fully pinned with [uv](https://docs.astral.sh/uv/)
(Python 3.12, see `.python-version`):

```bash
git clone --recurse-submodules https://github.com/daichi0812/babyloop.git
cd babyloop
uv sync
```

The official evaluation pipeline is vendored as a submodule pinned to a fixed
commit (`third_party/evaluation-pipeline`); all four cells are evaluated with
the same revision. If you skipped `--recurse-submodules`, run
`git submodule update --init --recursive`.

## Data

Text is the official BabyLM-2026-Strict corpus (100M words); the multimodal
cells use a deterministic 50M/50M word mixture of that corpus and the official
BabyLM multimodal release (OSF `ad7qg`: Localized Narratives + CC3M captions
with precomputed frozen DINOv2 features; raw images are never used).

```bash
# text-only corpus (Hugging Face)
uv run python scripts/download_data.py

# multimodal captions + DINOv2 features (OSF ad7qg)
uv run --with osfclient python scripts/download_mm_data.py --osf-dir data/mm/osf
uv run python scripts/precompute_vision.py --feature-source official --official-dir data/mm/osf

# preprocess (trains/reuses the frozen 16k BPE tokenizer, packs 512-token blocks)
uv run python scripts/preprocess.py experiment=std_text   # text cells
uv run python scripts/preprocess.py experiment=std_mm     # multimodal cells
```

Word-budget accounting (whitespace words, captions count against the same
100M budget) is enforced by `WordBudgetTracker`
(`src/babyloop/data/word_budget.py`, unit-tested). The assembled multimodal
mixture, its measured domain composition, and all provenance manifests are
published in the
[corpus repository](https://huggingface.co/datasets/daichi812/babyloop-mm-corpus).

## Training

Training is *words-seen driven*: checkpoints and the LR schedule are indexed
by words seen, not steps, up to a 1B-word horizon (~10 passes over the 100M
budget). Checkpoints are saved in HF-compatible form at the 28 official
milestones. bf16 on a single A100 is assumed.

```bash
uv run python scripts/train.py experiment=loop_mm             # one condition
uv run python scripts/train.py experiment=std_text train.seed=43
```

## Evaluation

```bash
uv run python scripts/evaluate.py experiment=loop_mm +checkpoint=<path>
```

`scripts/push_to_hub.py` pushes the final model and milestone checkpoints in
the challenge's submission format (loadable with
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`).

## Tests

```bash
uv run pytest
```

Key invariants: exact `K=1` equivalence to a standard Transformer, official
word-budget rule compliance, and words-seen LR-schedule correctness.

## Layout

```
babyloop/
├── configs/              # Hydra configs (experiment/ = one file per condition)
├── src/babyloop/
│   ├── models/           # LoopedTransformer, vision encoder, fusion strategies
│   ├── data/             # preprocessing, word-budget accounting, dataloaders
│   ├── training/         # trainer, words-seen checkpoint manager
│   ├── evaluation/       # wrapper around the pinned official pipeline
│   └── analysis/         # probing / analysis utilities
├── scripts/              # entry points (download / preprocess / train / evaluate / push)
├── tests/
└── third_party/evaluation-pipeline   # official eval pipeline (pinned submodule)
```

## Paper

The accompanying paper is under double-blind review at the BabyLM 2026
workshop; a citation will be added upon publication.

## License

[MIT](LICENSE)
