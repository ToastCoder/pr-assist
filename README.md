# pr-assist

LoRA fine-tuned [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) for pull request code review.

Default adapter: [toastcoder/pr-review-qwen-lora](https://huggingface.co/toastcoder/pr-review-qwen-lora)

## Project layout

```
pr-assist/
├── src/
│   ├── inference.py   # CLI and library API for review generation
│   ├── train.py       # LoRA fine-tuning script
│   ├── prompts.py     # PR prompt templates
│   └── chunking.py    # Split large diffs by file
├── scripts/
│   ├── create_subset.py
│   └── verify_adapter.py
└── notebooks/         # Exploratory training notebooks
```

Legacy experiments and local checkpoints live under `archive/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run inference

Interactive mode:

```bash
python -m src.inference
```

Review a pull request from flags:

```bash
python -m src.inference \
  --title "Add JWT auth middleware" \
  --description "Protect user routes with JWT tokens." \
  --diff-file path/to/changes.diff
```

Review a large diff in file chunks:

```bash
python -m src.inference \
  --title "Refactor auth module" \
  --description "Split auth into middleware and helpers." \
  --diff-file path/to/large.diff \
  --chunk
```

Pipe a full prompt on stdin:

```bash
cat prompt.txt | python -m src.inference
```

## Training

Create a small local dataset:

```bash
python scripts/create_subset.py --size 1000 --output data/processed/train_small.jsonl
```

Fine-tune LoRA:

```bash
python -m src.train \
  --data-files data/processed/train_small.jsonl \
  --output-dir models/pr-assist-qwen
```

## Smoke test

```bash
python scripts/verify_adapter.py
```

