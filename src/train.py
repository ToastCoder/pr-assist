# pr-assist
# train.py

import argparse
import warnings

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Suppress LibreSSL noise that clutters logs on macOS
warnings.filterwarnings("ignore", message=".*LibreSSL.*")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_DIR = "models/pr-assist-qwen"
DEFAULT_DATA_FILES = "data/processed/train_small.jsonl"
DEFAULT_MAX_LENGTH = 1024


def load_data(data_files: str):
    return load_dataset("json", data_files=data_files)["train"]


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        low_cpu_mem_usage=True,
    )

    return model, tokenizer


def format_sample(sample, tokenizer):
    # Utility for inspecting how a sample looks after ChatML formatting
    messages = [
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["output"]},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False)


def tokenize_sample(sample, tokenizer, max_length: int):
    user_messages = [{"role": "user", "content": sample["instruction"]}]
    full_messages = user_messages + [
        {"role": "assistant", "content": sample["output"]}
    ]

    prompt_text = tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)

    prompt_token_count = len(
        tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    )

    tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
    )

    # Mask the user prompt tokens (-100) so the loss is computed
    # only on the assistant response tokens
    labels = tokens["input_ids"].copy()
    prompt_token_count = min(prompt_token_count, len(labels))
    labels[:prompt_token_count] = [-100] * prompt_token_count
    tokens["labels"] = labels

    return tokens


def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        # Fine-tune all four attention projections for better adaptation
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_training_args(output_dir: str, num_train_epochs: int):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Effective batch size = 4
        learning_rate=1e-4,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )


def prepare_dataset(dataset, tokenizer, max_length: int):
    return dataset.map(
        tokenize_sample,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=dataset.column_names,
    )


def prepare_model(model):
    model = get_peft_model(model, get_lora_config())
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # Required when gradient checkpointing is on
    return model


def create_trainer(model, tokenizer, dataset, output_dir: str, num_train_epochs: int):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return Trainer(
        model=model,
        args=get_training_args(output_dir, num_train_epochs),
        train_dataset=dataset,
        data_collator=data_collator,
    )


def print_dataset_stats(dataset, max_length: int):
    lengths = [len(row["input_ids"]) for row in dataset]
    truncated = sum(1 for length in lengths if length >= max_length)

    print(f"Average Tokens: {sum(lengths) / len(lengths):.2f}")
    print(f"Max Tokens: {max(lengths)}")
    print(f"Min Tokens: {min(lengths)}")
    print(f"Truncated Samples: {truncated}/{len(lengths)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen with LoRA for pull request review."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Base model name or path",
    )
    parser.add_argument(
        "--data-files",
        type=str,
        default=DEFAULT_DATA_FILES,
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints and final adapter",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading dataset...")
    dataset = load_data(args.data_files)
    print(f"Samples: {len(dataset)}")

    print("Loading model...")
    model, tokenizer = load_model(args.model)
    print(
        f"Loaded {model.config.model_type} "
        f"({model.num_parameters():,} parameters)"
    )

    print("Applying LoRA...")
    model = prepare_model(model)
    model.print_trainable_parameters()

    print("Tokenizing dataset...")
    tokenized_dataset = prepare_dataset(dataset, tokenizer, args.max_length)
    print_dataset_stats(tokenized_dataset, args.max_length)

    print("Creating trainer...")
    trainer = create_trainer(
        model,
        tokenizer,
        tokenized_dataset,
        args.output_dir,
        args.epochs,
    )

    print("Training started...")
    trainer.train()

    final_dir = f"{args.output_dir}/final"
    print(f"Saving model to {final_dir}...")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done")


if __name__ == "__main__":
    main()
