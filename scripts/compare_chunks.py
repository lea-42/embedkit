"""
Interactive chunk comparison — step through two extractor outputs and align manually.

Usage:
    python scripts/compare_chunks.py tests/data/foo.pdf.openai.json tests/data/foo.pdf.pymupdf.json

Controls:
    Enter   advance both
    l       advance left only
    r       advance right only
    b       go back both
    q       quit
"""
import json
import sys
from pathlib import Path


def _label(path: str) -> str:
    return Path(path).stem


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = (current + " " + word).lstrip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _render_chunk(chunk: dict | None, label: str, idx: int, total: int) -> list[str]:
    if chunk is None:
        return [f"  [{label}] — end of file —"]
    crumbs = " > ".join(chunk.get("section_breadcrumbs") or []) or "(none)"
    page = chunk.get("page_number", "?")
    lines = [f"  [{label}]  chunk {idx}/{total - 1}  p{page}  {crumbs}"]
    lines.append("")
    for line in _wrap(chunk["text"], 90):
        lines.append(f"  {line}")
    return lines


def _print_pair(
    left: list[dict], right: list[dict],
    li: int, ri: int,
    left_label: str, right_label: str,
) -> None:
    lc = left[li] if li < len(left) else None
    rc = right[ri] if ri < len(right) else None

    print("\033[2J\033[H", end="")  # clear screen
    print("━" * 100)
    print(f"  controls:  Enter=advance both  l=left only  r=right only  b=back both  q=quit")
    print("━" * 100)

    l_lines = _render_chunk(lc, left_label, li, len(left))
    r_lines = _render_chunk(rc, right_label, ri, len(right))

    print()
    for line in l_lines:
        print(line)
    print()
    print("  " + "─" * 96)
    print()
    for line in r_lines:
        print(line)
    print()
    print("━" * 100)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: compare_chunks.py <left.json> <right.json>")
        sys.exit(1)

    left = json.loads(Path(sys.argv[1]).read_text())
    right = json.loads(Path(sys.argv[2]).read_text())
    left_label = _label(sys.argv[1])
    right_label = _label(sys.argv[2])

    li, ri = 0, 0

    while True:
        _print_pair(left, right, li, ri, left_label, right_label)

        if li >= len(left) and ri >= len(right):
            print("  Both files exhausted.")
            break

        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "q":
            break
        elif cmd == "l":
            li = min(li + 1, len(left))
        elif cmd == "r":
            ri = min(ri + 1, len(right))
        elif cmd == "b":
            li = max(li - 1, 0)
            ri = max(ri - 1, 0)
        else:
            # Enter or anything else — advance both
            li = min(li + 1, len(left))
            ri = min(ri + 1, len(right))


if __name__ == "__main__":
    main()
