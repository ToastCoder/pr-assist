# src/training/train.py

# pyrefly: ignore [missing-import]
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

def load_data():
    return load_dataset(
        "json",
        data_files="data/processed/train_small.jsonl"
    )["train"]

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto"
    )

    return model, tokenizer

def main():
    dataset = load_data()

    print(f"Dataset size: {len(dataset)}")

    model, tokenizer = load_model()

    print("Model loaded successfully")

if __name__ == "__main__":
    main()