# CENG534 — Building GPT-2 from Scratch

A from-scratch reimplementation of GPT-2 (our own modules, loaded with Hugging Face's pretrained
weights) applied to three downstream NLP tasks, plus a parameter-efficient fine-tuning (LoRA)
extension. Term project for **CENG534 Natural Language Processing**, İzmir Institute of Technology
(İYTE), built on Stanford CS224N's *Default Final Project: Build GPT-2*.

**Authors:** Hakan Boğan, İlker Yayalar

## Overview

The project has three parts:

- **Part 1 — Backbone and optimizer.** Implement GPT-2 (causal multi-head self-attention, the
  pre-LayerNorm block, learned token/position embeddings, and a weight-tied language-modeling head)
  and the AdamW optimizer from scratch, then transfer the official GPT-2 weights into our modules
  and verify numerical equivalence with the reference implementations.
- **Part 2 — Downstream tasks.** Adapt the validated model to five-way sentiment classification
  (SST, CFIMDB), Quora paraphrase detection, and Shakespearean sonnet generation.
- **Part 3 — Extension.** Low-Rank Adaptation (LoRA) on the attention query/key/value projections,
  with three initialization schemes (`default`, `weight_dist`, `svd`).

## What is implemented

- **From-scratch GPT-2** (`src/models/gpt2.py`): scaled dot-product attention with a causal mask
  (`src/modules/attention.py`), the pre-LN transformer block (`src/modules/gpt2_layer.py`), and a
  weight-tied output head (`logits = h E^T`). `from_pretrained` remaps Hugging Face's fused `c_attn`
  weight into three separate, transposed linear projections (Q/K/V).
- **Custom AdamW** (`src/optimizer.py`) with decoupled weight decay.
- **Task heads:** sentiment and paraphrase use the last non-padding token representation with a
  linear classifier; sonnet generation uses the weight-tied head with shifted cross-entropy and
  nucleus (top-`p`) sampling.
- **LoRA adapters** (`src/modules/attention.py`): `ΔW = (α/r)·BA` on the Q/K/V projections of all
  12 layers, with `r=8`, `α=16` and the three initialization schemes above.

## Repository layout

```text
our_code/
├── classifier.py            # sentiment classification (SST, CFIMDB)
├── paraphrase_detection.py  # Quora paraphrase detection
├── sonnet_generation.py     # GPT-2 sonnet generation
├── src/                     # model, optimizer, datasets, evaluation, LoRA stats
│   ├── models/gpt2.py        modules/attention.py  modules/gpt2_layer.py
│   ├── optimizer.py          datasets.py           evaluation.py
│   └── config.py  utils.py  log_experiments.py  lora_attention.py  distribution_metrics.py
├── scripts/                 # sanity_check.py, optimizer_test.py, prepare_submit.py, test_lora_init.py
├── others/                  # env.yml, setup.sh, LICENSE
├── data/                    # SST / CFIMDB / Quora / sonnet splits
├── predictions/             # submission-style output files
└── results/                 # logs, metric CSVs, figures, sonnet checkpoints
```

See [`_PROJECT_STRUCTURE.md`](_PROJECT_STRUCTURE.md) for the full file-by-file map.

## Setup

Create the conda environment (Python 3.8) and activate it:

```bash
conda env create -f others/env.yml
conda activate cs224n_dfp
```

Run every command **from the repository root** (`our_code/`): the scripts use relative paths
(`data/...`, `predictions/...`) and only work when that is the working directory.

## Part 1 — correctness checks

These are the project's tests; there is no separate test runner.

```bash
python scripts/sanity_check.py     # our GPT-2 hidden states vs Hugging Face GPT-2 (atol 1e-1)
python scripts/optimizer_test.py   # our AdamW step vs the reference tensor (atol 1e-6)
```

## Part 2 & 3 — running the tasks

Add `--use_gpu` to train on a GPU. Tune `--batch_size` to your GPU memory to avoid out-of-memory
errors (Quora is the heaviest task).

```bash
# Sentiment classification
python classifier.py --use_gpu --fine-tune-mode last-linear-layer   # frozen backbone, head only
python classifier.py --use_gpu --fine-tune-mode full-model          # all parameters

# Paraphrase detection (Quora)
python paraphrase_detection.py --use_gpu --fine_tune_mode full
python paraphrase_detection.py --use_gpu --fine_tune_mode lora \
    --lora_r 8 --lora_alpha 16 --lora_init_method weight_dist

# Sonnet generation
python sonnet_generation.py --use_gpu --fine_tune_mode full
python sonnet_generation.py --use_gpu --fine_tune_mode lora \
    --lora_r 8 --lora_alpha 16 --lora_init_method svd
```

`--lora_init_method` accepts `default`, `weight_dist`, or `svd`. The full set of copy-ready commands
(full / last-layer / LoRA, GPU and CPU) is in [`_RUN_COMMANDS.md`](_RUN_COMMANDS.md). Sonnet
checkpoints are written to `results/sonnet_generation_checkpoints/` (gitignored).

## Results (development set)

All numbers are measured on the **development split only**; test labels are never used locally, in
line with the dataset usage policy.

| Task        | Dataset      | Metric             | Full fine-tune   | Frozen / probe |
|-------------|--------------|--------------------|------------------|----------------|
| Sentiment   | SST (5-way)  | accuracy           | 0.5077           | 0.4550         |
| Sentiment   | CFIMDB       | accuracy           | 0.9837           | 0.8776         |
| Paraphrase  | Quora        | accuracy / macro-F1| 0.8978 / 0.8912  | —              |
| Sonnet gen. | Sonnets      | chrF / BLEU        | 40.95 / 23.75    | —              |

**LoRA extension (Part 3).** Training only 0.44M parameters (0.36% of the model), LoRA with the
`weight_dist` initialization reaches chrF 40.09 on the held-out sonnets — 97.9% of the full
fine-tuning score — while every LoRA variant outperforms last-layer fine-tuning at a fraction of its
trainable-parameter cost.

## Preparing a submission

```bash
python scripts/prepare_submit.py   # zips the .py files, predictions/, and modules for Gradescope
```

## Acknowledgements and license

Adapted from Stanford CS224N's *Default Final Project: Build GPT-2*. Parts of the code are from the
[Hugging Face `transformers`](https://github.com/huggingface/transformers) library
([Apache License 2.0](others/LICENSE)).
