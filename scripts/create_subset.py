# pr-assist
# create_subset.py

import argparse
from pathlib import Path

from datasets import load_dataset

DEFAULT_DATASET = "opencsg/PR_review_deepseek"
DEFAULT_OUTPUT = "data/processed/train_small.jsonl"
DEFAULT_SAMPLE_SIZE = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a smaller JSONL subset from the PR review dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of training samples to export",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset)
    subset = dataset["train"].select(range(args.size))

    print(f"Exporting {len(subset)} samples to {output_path}")
    subset.to_json(str(output_path))
    print("Done")


if __name__ == "__main__":
    main()
