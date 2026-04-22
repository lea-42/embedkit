"""
Show text present in one extractor output but missing in the other.

Usage:
    python scripts/diff_chunks.py tests/data/foo.pdf.openai.json tests/data/foo.pdf.pymupdf.json
"""
import difflib
import json
import sys
from pathlib import Path


def _load_text(path: str) -> str:
    chunks = json.loads(Path(path).read_text())
    return "\n".join(c["text"] for c in chunks)


def _label(path: str) -> str:
    return Path(path).stem


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: diff_chunks.py <left.json> <right.json>")
        sys.exit(1)

    left_text = _load_text(sys.argv[1])
    right_text = _load_text(sys.argv[2])
    left_label = _label(sys.argv[1])
    right_label = _label(sys.argv[2])

    left_lines = left_text.splitlines()
    right_lines = right_text.splitlines()

    diff = list(difflib.unified_diff(
        left_lines,
        right_lines,
        fromfile=left_label,
        tofile=right_label,
        lineterm="",
    ))

    if not diff:
        print("No differences found.")
        return

    only_left: list[str] = []
    only_right: list[str] = []

    for line in diff:
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("-"):
            only_left.append(line[1:].strip())
        elif line.startswith("+"):
            only_right.append(line[1:].strip())

    only_left = [l for l in only_left if l]
    only_right = [l for l in only_right if l]

    if only_left:
        print(f"\n{'━' * 80}")
        print(f"  Only in {left_label} ({len(only_left)} lines):")
        print(f"{'━' * 80}")
        for line in only_left:
            print(f"  - {line}")

    if only_right:
        print(f"\n{'━' * 80}")
        print(f"  Only in {right_label} ({len(only_right)} lines):")
        print(f"{'━' * 80}")
        for line in only_right:
            print(f"  + {line}")

    print(f"\n  Summary: {len(only_left)} lines only in {left_label}, {len(only_right)} lines only in {right_label}")


if __name__ == "__main__":
    main()
