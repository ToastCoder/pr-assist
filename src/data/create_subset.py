# pyrefly: ignore [missing-import]
from datasets import load_dataset

dataset = load_dataset("opencsg/PR_review_deepseek")

small_train = dataset["train"].select(range(1000))

print(len(small_train))

small_train.to_json(
    "data/processed/train_small.jsonl"
)