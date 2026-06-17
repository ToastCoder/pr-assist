# src/training/train.py

import warnings

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

warnings.filterwarnings(
    "ignore",
    message=".*LibreSSL.*"
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "models/pr-assist-qwen"


def load_data():
    return load_dataset(
        "json",
        data_files="data/processed/train_small.jsonl",
    )["train"]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        low_cpu_mem_usage=True,
    )

    return model, tokenizer


def format_sample(sample, tokenizer):
    messages = [
        {
            "role": "user",
            "content": sample["instruction"],
        },
        {
            "role": "assistant",
            "content": sample["output"],
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )


def tokenize_sample(sample, tokenizer):
    formatted_text = format_sample(
        sample,
        tokenizer,
    )

    tokens = tokenizer(
        formatted_text,
        truncation=True,
        max_length=1024,
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=1,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

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


def prepare_dataset(dataset, tokenizer):
    return dataset.map(
        tokenize_sample,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        remove_columns=dataset.column_names,
    )


def prepare_model(model):
    lora_config = get_lora_config()

    model = get_peft_model(
        model,
        lora_config,
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model


def create_trainer(
    model,
    tokenizer,
    dataset,
):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    return Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=dataset,
        data_collator=data_collator,
    )


def print_dataset_stats(dataset):
    lengths = [
        len(row["input_ids"])
        for row in dataset
    ]

    truncated = sum(
        1
        for length in lengths
        if length >= 512
    )

    print(
        f"Average Tokens: "
        f"{sum(lengths) / len(lengths):.2f}"
    )

    print(
        f"Max Tokens: "
        f"{max(lengths)}"
    )

    print(
        f"Min Tokens: "
        f"{min(lengths)}"
    )

    print(
        f"Truncated Samples: "
        f"{truncated}/{len(lengths)}"
    )


def main():
    print("Loading dataset...")
    dataset = load_data()

    print(
        f"Samples: {len(dataset)}"
    )

    print("Loading model...")
    model, tokenizer = load_model()

    print(
        f"Loaded {model.config.model_type} "
        f"({model.num_parameters():,} parameters)"
    )

    print("Applying LoRA...")
    model = prepare_model(model)

    model.print_trainable_parameters()

    print("Tokenizing dataset...")
    tokenized_dataset = prepare_dataset(
        dataset,
        tokenizer,
    )

    print_dataset_stats(
        tokenized_dataset
    )

    print("Creating trainer...")
    trainer = create_trainer(
        model,
        tokenizer,
        tokenized_dataset,
    )

    print("Training started...")
    trainer.train()

    print("Saving model...")

    model.save_pretrained(
        f"{OUTPUT_DIR}/final"
    )

    tokenizer.save_pretrained(
        f"{OUTPUT_DIR}/final"
    )

    print("Done")


if __name__ == "__main__":
    main()