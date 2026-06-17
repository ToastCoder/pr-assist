# pyrefly: ignore [missing-import]
from datasets import load_dataset
from src.training.prompts import format_sample

dataset = load_dataset(
    "json",
    data_files="data/processed/train_small.jsonl"
)["train"]

print(format_sample(dataset[0]))