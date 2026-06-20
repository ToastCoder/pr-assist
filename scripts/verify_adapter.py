# pr-assist
# verify_adapter.py

import argparse

from src.inference import BASE_MODEL, DEFAULT_ADAPTER, generate_review, load_model
from src.prompts import sample_review_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke test the PR review LoRA adapter."
    )
    parser.add_argument(
        "--base",
        type=str,
        default=BASE_MODEL,
        help="Base model name or path",
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
        "--max-new-tokens",
        type=int,
        default=300,
        help="Maximum tokens to generate",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading model...")
    model, tokenizer = load_model(
        base_model_name=args.base,
        adapter_name=args.adapter,
        device=args.device,
    )

    prompt = sample_review_prompt()
    print("Generating sample review...\n")
    response = generate_review(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print("Response:\n")
    print(response)


if __name__ == "__main__":
    main()
