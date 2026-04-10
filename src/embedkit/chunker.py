import re


def _clean_formatting(text: str) -> str:
    """Strip markdown inline formatting: bold, italic, strikethrough."""
    text = re.sub(r"~~(.*?)~~", r"\1", text)   # strikethrough
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)    # italic
    text = re.sub(r"__(.*?)__", r"\1", text)    # bold underscore
    text = re.sub(r"_(.*?)_", r"\1", text)      # italic underscore
    return text


def _heading_level(line: str) -> int | None:
    m = re.match(r"^(#{1,6})\s", line)
    return len(m.group(1)) if m else None


def _numeric_depth(heading_text: str) -> int | None:
    """Return inferred depth from a leading numeric prefix, or None if not numbered.

    "2"     → 1  (0 dots → depth 1)
    "2.1"   → 2  (1 dot  → depth 2)
    "2.1.3" → 3  (2 dots → depth 3)
    """
    m = re.match(r"^(\d+(?:\.\d+)*)\s", heading_text)
    if not m:
        return None
    return m.group(1).count(".") + 1


def _is_page_separator(line: str) -> bool:
    return bool(re.match(r"^-{3,}\s*$", line))


def _is_table_line(line: str) -> bool:
    return line.startswith("|")


def _is_list_line(line: str) -> bool:
    return bool(re.match(r"^\s*[-*+]\s", line))


def chunk_markdown(md: str) -> list[dict]:
    """Parse pymupdf4llm markdown into chunks with breadcrumb tracking.

    Each chunk has: page_number, chunk_number, section_breadcrumbs, text.
    Tables and lists are kept as single chunks. Paragraphs are split on blank lines.
    Page number increments on pymupdf4llm page separators (---).
    """
    breadcrumb_stack: list[str] = []   # [(depth, text), ...] for numbered; reset on non-numbered
    breadcrumb_depths: list[int] = []
    page = 0
    chunks: list[dict] = []
    pending_lines: list[str] = []

    def flush(lines: list[str]) -> None:
        text = _clean_formatting("\n".join(lines).strip())
        if text:
            chunks.append({
                "page_number": page,
                "chunk_number": len(chunks),
                "section_breadcrumbs": list(breadcrumb_stack),
                "text": text,
            })

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        if _is_page_separator(line):
            flush(pending_lines)
            pending_lines = []
            page += 1
            i += 1
            continue

        level = _heading_level(line)
        if level is not None:
            flush(pending_lines)
            pending_lines = []
            heading_text = _clean_formatting(re.sub(r"^#{1,6}\s+", "", line).strip())
            depth = _numeric_depth(heading_text)
            if depth is None:
                # Non-numbered: flat single breadcrumb, reset stack
                breadcrumb_stack[:] = [heading_text]
                breadcrumb_depths[:] = []
            else:
                # Numbered: if stack has non-numbered content (depths empty but stack not),
                # clear it first so numbered sections don't nest under non-numbered ones
                if not breadcrumb_depths and breadcrumb_stack:
                    breadcrumb_stack.clear()
                # Pop stack entries at same depth or deeper, then push
                while breadcrumb_depths and breadcrumb_depths[-1] >= depth:
                    breadcrumb_stack.pop()
                    breadcrumb_depths.pop()
                breadcrumb_stack.append(heading_text)
                breadcrumb_depths.append(depth)
            i += 1
            continue

        if _is_table_line(line):
            flush(pending_lines)
            pending_lines = []
            table_lines = []
            while i < len(lines) and _is_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1
            flush(table_lines)
            continue

        if _is_list_line(line):
            flush(pending_lines)
            pending_lines = []
            list_lines = []
            while i < len(lines) and (lines[i].strip() == "" or _is_list_line(lines[i])):
                if lines[i].strip():
                    list_lines.append(lines[i])
                elif list_lines and i + 1 < len(lines) and _is_list_line(lines[i + 1]):
                    pass  # skip blank lines within list
                else:
                    break
                i += 1
            flush(list_lines)
            continue

        if line.strip() == "":
            flush(pending_lines)
            pending_lines = []
        else:
            pending_lines.append(line)

        i += 1

    flush(pending_lines)
    return _merge_chunks(chunks)


MIN_CHUNK_CHARS = 200


def _ends_with_colon(text: str) -> bool:
    return bool(re.search(r":-?\s*$", text.strip()))


def _same_context(a: dict, b: dict) -> bool:
    return a["page_number"] == b["page_number"] and a["section_breadcrumbs"] == b["section_breadcrumbs"]


def _merge_chunks(chunks: list[dict]) -> list[dict]:
    """Post-processing merges (only within same page and section):
    1. If a chunk ends with ':' or ':-', merge it with the next chunk.
    2. If a chunk is under MIN_CHUNK_CHARS, merge it with the next chunk.
    Chunk numbers are recomputed after merging.
    """
    if not chunks:
        return chunks

    merged: list[dict] = []
    i = 0
    while i < len(chunks):
        current = dict(chunks[i])
        while i + 1 < len(chunks) and _same_context(current, chunks[i + 1]) and (
            _ends_with_colon(current["text"]) or len(current["text"]) < MIN_CHUNK_CHARS
        ):
            current["text"] = current["text"] + "\n\n" + chunks[i + 1]["text"]
            i += 1
        merged.append(current)
        i += 1

    for idx, chunk in enumerate(merged):
        chunk["chunk_number"] = idx

    return merged
