# pyrefly: ignore [missing-import]
from datasets import load_dataset

dataset = load_dataset("opencsg/PR_review_deepseek")

sample = dataset["train"][0]

instruction = sample["instruction"]
output = sample["output"]

print("INPUT LENGTH:", len(instruction))
print("OUTPUT LENGTH:", len(output))