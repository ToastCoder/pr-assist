# src/data/stats.py

# pyrefly: ignore [missing-import]
from datasets import load_dataset

dataset = load_dataset("opencsg/PR_review_deepseek")

train = dataset["train"]

print(f"Samples: {len(train)}")

print(f"Average prompt length: {sum(train['prompt_len']) / len(train):.2f}")
print(f"Average response length: {sum(train['response_len']) / len(train):.2f}")

print(f"Max prompt length: {max(train['prompt_len'])}")
print(f"Max response length: {max(train['response_len'])}")

print(f"Max total length: {max(train['total_len'])}")