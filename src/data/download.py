# pr-assist
# src/data/download.py

# Importing Libraries
# pyrefly: ignore [missing-import]
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("opencsg/PR_review_deepseek")

print(dataset)

print("\nColumns:")

print(dataset["train"].column_names)

print("\nFirst Example:")

print(dataset["train"][0])