import logging
import re
from collections.abc import Callable

import pymupdf4llm

from docvec.chunker import chunk_markdown
from docvec.extractors.base import BaseExtractor
from docvec.extractors.models import PageTables

logger = logging.getLogger(__name__)

PICTURE_TABLE_MARKER = "Start of picture text"
MARKDOWN_TABLE_RE = re.compile(r"^\|.+\|")


def _scan_pages(md: str, match_fn: Callable[[str], bool]) -> list[int]:
    """Walk pymupdf4llm markdown line by line, return 1-based page numbers where match_fn fires."""
    matched = []
    current_page = 1
    for line in md.splitlines():
        m = re.search(r"page\.page_number=(\d+)", line)
        if m:
            current_page = int(m.group(1)) + 1
        elif match_fn(line) and current_page not in matched:
            matched.append(current_page)
    return sorted(matched)


def find_picture_table_pages(md: str) -> list[int]:
    """Return 1-based page numbers with image-rendered tables (garbled by pymupdf4llm).

    These are the pages to send to OpenAI for accurate table extraction.
    Pages with clean markdown tables (|...|) are already correctly extracted and excluded.
    """
    return _scan_pages(md, lambda line: PICTURE_TABLE_MARKER in line)


def find_table_pages(md: str) -> list[int]:
    """Return 1-based page numbers that contain any table (picture or markdown)."""
    return _scan_pages(md, lambda line: PICTURE_TABLE_MARKER in line or bool(MARKDOWN_TABLE_RE.match(line)))


# Matches the full picture-text block including start/end markers (may span multiple lines)
_PICTURE_BLOCK_RE = re.compile(
    r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*",
    re.DOTALL,
)


def _tables_to_markdown(page_tables: PageTables) -> str:
    """Render PageTables as markdown table(s), one per table separated by a blank line."""
    table_mds = []
    for table in page_tables.tables:
        t_parts = []
        if table.caption:
            t_parts.append(f"*{table.caption}*")
        if not table.cells:
            continue
        max_row = max(c.row for c in table.cells)
        max_col = max(c.col for c in table.cells)
        grid = [[""] * (max_col + 1) for _ in range(max_row + 1)]
        for c in table.cells:
            grid[c.row][c.col] = c.text
        for r_idx, row in enumerate(grid):
            t_parts.append("| " + " | ".join(row) + " |")
            if r_idx == 0:
                t_parts.append("| " + " | ".join("---" for _ in row) + " |")
        table_mds.append("\n".join(t_parts))
    return "\n\n".join(table_mds)


def _split_pages(md: str) -> list[tuple[int, str]]:
    """Split markdown into (1-based page_no, page_content) pairs.

    pymupdf4llm emits '--- end of page.page_number=N ---' after each page's content,
    so lines before that separator belong to page N.
    """
    pages = []
    current_lines: list[str] = []
    for line in md.splitlines(keepends=True):
        m = re.search(r"page\.page_number=(\d+)", line)
        if m:
            current_lines.append(line)
            pages.append((int(m.group(1)), "".join(current_lines)))
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        # trailing content after last separator (e.g. blank lines at EOF)
        pages.append((pages[-1][0] + 1 if pages else 1, "".join(current_lines)))
    return pages


def replace_picture_tables(md: str, table_results: dict[int, PageTables]) -> str:
    """Replace garbled picture-text blocks with clean markdown tables.

    table_results: dict mapping 1-based page_no → PageTables from OpenAI extraction.
    Each picture-text block is replaced with the corresponding table (block i → table i).
    Pages not in table_results are left unchanged.
    """
    pages = _split_pages(md)
    out = []
    for page_no, content in pages:
        if page_no in table_results:
            page_tables = table_results[page_no]
            table_mds = [_tables_to_markdown(PageTables(tables=[t])) for t in page_tables.tables]
            n_before = len(_PICTURE_BLOCK_RE.findall(content))
            counter = [0]

            def _replacer(m: re.Match) -> str:
                idx = counter[0]
                counter[0] += 1
                return table_mds[idx] if idx < len(table_mds) else ""

            content = _PICTURE_BLOCK_RE.sub(_replacer, content)
            logger.info("page %d: replaced %d picture-text block(s) with %d table(s)", page_no, n_before, len(page_tables.tables))
        out.append(content)
    return "".join(out)


def _promote_numbered_headings(md: str) -> str:
    """Add one '#' per dot in numbered heading prefixes like '2.1' or '2.1.3'.

    pymupdf4llm flattens all headings to '##'. For headings whose text starts
    with a numeric prefix (e.g. '2.1 Travel Exclusions'), this restores depth:
      '## 2 ...'     → '## 2 ...'      (0 dots, no change)
      '## 2.1 ...'   → '### 2.1 ...'   (1 dot, +1 #)
      '## 2.1.3 ...' → '#### 2.1.3 ..' (2 dots, +2 #)
    """
    def _replace(m: re.Match) -> str:
        prefix = m.group(2)
        full_text = m.group(1)
        extra = "#" * prefix.count(".")
        return f"##{extra} {full_text}"

    return re.sub(r"^## ((\d+(?:\.\d+)+)\s.*)$", _replace, md, flags=re.MULTILINE)


class PyMuPDFExtractor(BaseExtractor):
    """Fast, local extractor using pymupdf4llm. No API cost."""

    def __init__(self, max_pages: int | None = None) -> None:
        self.max_pages = max_pages

    def extract_raw(self, pdf_path: str) -> str:
        """Return the cleaned markdown string before chunking."""
        kwargs: dict = {"use_ocr": False, "page_separators": True}
        if self.max_pages is not None:
            kwargs["pages"] = list(range(self.max_pages))
        md = pymupdf4llm.to_markdown(pdf_path, **kwargs)
        md = re.sub(r"^(- )n ", r"\1", md, flags=re.MULTILINE)
        return _promote_numbered_headings(md)

    def extract(self, pdf_path: str) -> list[dict]:
        return chunk_markdown(self.extract_raw(pdf_path))
