# pr-assist
# chunking.py

from __future__ import annotations

from .prompts import format_pr_prompt


def split_diff_by_file(diff: str) -> list[str]:
    """Splits a unified diff into per-file chunks."""
    if not diff.strip():
        return []

    chunks = []
    current: list[str] = []

    for line in diff.splitlines(keepends=True):
        # "diff --git" is git's per-file separator in unified diffs
        if line.startswith("diff --git ") and current:
            chunks.append("".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("".join(current))

    if not chunks:
        return [diff]

    return chunks


def build_chunked_prompts(
    title: str,
    description: str,
    diff: str,
    max_chunks: int | None = None,
) -> list[str]:
    """Builds one review prompt per diff file chunk."""
    chunks = split_diff_by_file(diff)

    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    if len(chunks) <= 1:
        return [format_pr_prompt(title, description, diff)]

    prompts = []
    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        # Let the model know this is file N of M so it can contextualise
        chunk_description = (
            f"{description}\n\n"
            f"(Reviewing file chunk {index} of {total} from this pull request.)"
        )
        prompts.append(format_pr_prompt(title, chunk_description, chunk))

    return prompts
