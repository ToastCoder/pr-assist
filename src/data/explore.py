# pr-assist
# src/data/explore.py


# pyrefly: ignore [missing-import]
from datasets import load_dataset

dataset = load_dataset("opencsg/PR_review_deepseek")

sample = dataset["train"][0]

print("=" * 50)
print("INSTRUCTION")
print("=" * 50)
print(sample["instruction"][:1000])

print("\n") 

print("=" * 50)
print("OUTPUT")
print("=" * 50)
print(sample["output"][:1000])