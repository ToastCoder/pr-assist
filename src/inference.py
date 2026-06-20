# pr-assist
# inference.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chunking import build_chunked_prompts
from src.prompts import format_pr_prompt

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = "toastcoder/pr-review-qwen-lora"


def get_device(device_arg: str | None = None) -> str:
    """Detects the best available accelerator device."""
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
    device: str | None = None,
):
    """Loads the tokenizer, base model and mounts the PEFT adapter."""
    if adapter_name is None:
        adapter_name = DEFAULT_ADAPTER

    target_device = get_device(device)
    print(f"Loading base model: {base_model_name}")
    print(f"Loading adapter: {adapter_name}")
    print(f"Using device: {target_device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Qwen doesn't have a native pad token; reuse eos to avoid issues
    tokenizer.pad_token = tokenizer.eos_token

    if target_device in ["cuda", "mps"]:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype="auto", device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype="auto"
        ).to(target_device)

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
    max_input_tokens: int | None = None,
) -> str:
    """Generates review comments for a given instruction prompt."""
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    tokenize_kwargs = {"return_tensors": "pt"}
    if max_input_tokens is not None:
        tokenize_kwargs["truncation"] = True
        tokenize_kwargs["max_length"] = max_input_tokens

    inputs = tokenizer(text, **tokenize_kwargs).to(model.device)

    # Only include temperature when sampling (ignored in greedy mode)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

        # Strip the input prefix so we only return the model's answer
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()


def generate_reviews_for_pr(
    model,
    tokenizer,
    title: str,
    description: str,
    diff: str,
    chunk: bool = False,
    max_chunks: int | None = None,
    **generate_kwargs,
) -> list[str]:
    """Generates one or more review responses for a pull request."""
    if chunk:
        prompts = build_chunked_prompts(
            title, description, diff, max_chunks=max_chunks
        )
    else:
        prompts = [format_pr_prompt(title, description, diff)]

    return [
        generate_review(model, tokenizer, prompt, **generate_kwargs)
        for prompt in prompts
    ]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_multiline_input() -> str:
    """Reads multiline input from the user until END keyword."""
    print("\nPaste your prompt below.")
    print("Type 'END' on a new line when you are finished.\n")

    lines = []

    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break

    return "\n".join(lines)


def resolve_prompt(args) -> str | None:
    if args.prompt_file:
        return read_text_file(Path(args.prompt_file))

    if args.diff_file or args.title or args.description or args.diff:
        title = args.title or ""
        description = args.description or ""
        if args.diff_file:
            diff = read_text_file(Path(args.diff_file))
        else:
            diff = args.diff or ""
        return format_pr_prompt(title, description, diff)

    if not sys.stdin.isatty():
        return sys.stdin.read()

    return None


def parse_args():
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
    parser.add_argument(
        "--title", type=str, default=None, help="Pull request title"
    )
    parser.add_argument(
        "--description", type=str, default=None, help="Pull request description"
    )
    parser.add_argument(
        "--diff", type=str, default=None, help="Pull request diff text"
    )
    parser.add_argument(
        "--diff-file", type=str, default=None, help="Path to a diff file"
    )
    parser.add_argument(
        "--prompt-file", type=str, default=None, help="Path to a full prompt file"
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Split large diffs by file and review each chunk",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of diff chunks to review",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Maximum input tokens before truncation",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable sampling during generation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.sample,
        "max_input_tokens": args.max_input_tokens,
    }

    print("Loading model and adapter...")
    model, tokenizer = load_model(
        base_model_name=args.base, adapter_name=args.adapter, device=args.device
    )
    print("Model loaded successfully.")

    one_shot_prompt = resolve_prompt(args)
    if one_shot_prompt is not None:
        if args.chunk and (args.diff_file or args.diff):
            title = args.title or ""
            description = args.description or ""
            if args.diff_file:
                diff = read_text_file(Path(args.diff_file))
            else:
                diff = args.diff or ""
            responses = generate_reviews_for_pr(
                model,
                tokenizer,
                title,
                description,
                diff,
                chunk=True,
                max_chunks=args.max_chunks,
                **generate_kwargs,
            )
            for index, response in enumerate(responses, start=1):
                if len(responses) > 1:
                    print(f"\n--- Review {index}/{len(responses)} ---\n")
                print(response)
        else:
            response = generate_review(
                model, tokenizer, one_shot_prompt, **generate_kwargs
            )
            print(response)
        return

    while True:
        try:
            prompt = read_multiline_input()

            if not prompt.strip():
                continue
            if prompt.strip().lower() in ["exit", "quit"]:
                break

            print("\nGenerating review...\n")
            response = generate_review(
                model, tokenizer, prompt, **generate_kwargs
            )
            print("Response:\n", response)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
