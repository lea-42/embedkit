import re
from pathlib import Path

import pymupdf4llm

from docvec.chunker import chunk_markdown


def _promote_numbered_headings(md: str) -> str:
    """Add one '#' per dot in numbered heading prefixes like '2.1' or '2.1.3'.

    pymupdf4llm flattens all headings to '##'. For headings whose text starts
    with a numeric prefix (e.g. '2.1 Travel Exclusions'), this restores depth:
      '## 2 ...'     → '## 2 ...'      (0 dots, no change)
      '## 2.1 ...'   → '### 2.1 ...'   (1 dot, +1 #)
      '## 2.1.3 ...' → '#### 2.1.3 ..' (2 dots, +2 #)
    """
    def _replace(m: re.Match) -> str:
        prefix = m.group(2)   # e.g. '2.1'
        full_text = m.group(1)  # full heading text e.g. '2.1 Travel Exclusions'
        extra = "#" * prefix.count(".")
        return f"##{extra} {full_text}"

    return re.sub(r"^## ((\d+(?:\.\d+)+)\s.*)$", _replace, md, flags=re.MULTILINE)


def extract(pdf_path: str) -> list[dict]:
    """Extract a PDF and return chunks. No side effects."""
    md = pymupdf4llm.to_markdown(pdf_path, use_ocr=False, page_separators=True)
    md = _promote_numbered_headings(md)
    return chunk_markdown(md)
