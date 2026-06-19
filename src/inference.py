# pr-assist
# src/inference.py

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = "toastcoder/pr-review-qwen-lora"


def get_device(device_arg: str = None) -> str:
    """Detects the best available accelerator device."""

    # Check if user has provided a device
    if device_arg is not None:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    base_model_name: str = BASE_MODEL,
    adapter_name: str = DEFAULT_ADAPTER,
    device: str = None,
):
    """Loads the tokenizer, base model and mounts the PEFT Adapter."""
    if adapter_name is None:
        adapter_name = DEFAULT_ADAPTER

    target_device = get_device(device)
    print(f"Loading base model: {base_model_name}")
    print(f"Loading adapter: {adapter_name}")
    print(f"Using device: {target_device}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    if target_device in ["cuda", "mps"]:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype="auto", device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype="auto"
        ).to(target_device)

    # Load Peft Adapter
    model = PeftModel.from_pretrained(base_model, adapter_name)
    model.eval()

    return model, tokenizer


def generate_review(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """Generates review comments for a given instruction prompt."""

    # Format Prompt
    messages = [{"role": "user", "content": prompt}]

    # Convert Prompt to ChatML Format
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize Input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Generate Response
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Remove Input Prompt from Output
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]

        # Decode and Return Response
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()


def format_pr_prompt(title: str, description: str, diff: str) -> str:
    """Formats pull request information into a structured prompt."""

    prompt = (
        "Review the following pull request and provide detailed feedback.\n\n"
        f"PR Title:\n{title}\n\n"
        f"PR Description:\n{description}\n\n"
        f"Code Changes:\n\n{diff}"
    )
    return prompt


def read_multiline_input() -> str:
    """Reads multiline input from the user until END keyword."""

    print("\nPaste your prompt below.")
    print("Type 'END' on a new line when you are finished.\n")

    lines = []

    # Read lines until END keyword
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)


def main():
    """CLI driver for running model inference interactive mode."""

    # Argument Parser
    parser = argparse.ArgumentParser(
        description="Run pull request review inference using Qwen + LoRA."
    )
    parser.add_argument(
        "--base", type=str, default=BASE_MODEL, help="Base model name or path"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_ADAPTER,
        help="LoRA adapter name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device (cuda, mps, cpu)",
    )

    args = parser.parse_args()

    # Load model and adapter
    print("Loading model and the adapter")
    model, tokenizer = load_model(
        base_model_name=args.base, adapter_name=args.adapter, device=args.device
    )

    print("Model loaded successfully.")

    # Interactive Loop
    while True:
        try:
            # Read Prompt
            prompt = read_multiline_input()

            # Exit Conditions
            if not prompt.strip():
                continue
            if prompt.strip().lower() in ["exit", "quit"]:
                break
            print("\nGenerating Review...\n")
            response = generate_review(model, tokenizer, prompt)
            print("Response:\n", response)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
