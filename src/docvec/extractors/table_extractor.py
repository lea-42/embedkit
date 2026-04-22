"""OpenAI-based table extractor for picture-text table pages.

Sends individual pages in parallel to OpenAI and returns structured table cells.
Only used for pages where pymupdf4llm emits garbled 'Start of picture text' blocks.
"""
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
import instructor
from openai import OpenAI

from docvec.extractors.models import PageTables, Table, TableCell  # noqa: F401 — re-exported

logger = logging.getLogger(__name__)


PROMPT = """\
This is a single page from a French insurance document.

Extract ONLY the tables on this page. For each table:
- caption: a short description of what the table is about (optional, if obvious from context)
- cells: every cell as (row, col, text), both 0-based. Row 0 is always the header row.
- Include every cell, use empty string for empty cells.
- Preserve the exact text from the table — do not translate, summarise, or omit.

Do not extract any prose or body text. If there are no tables, return an empty list.
"""


def _slice_page(pdf_path: str, page_no: int) -> bytes:
    """Return a single 1-based page as PDF bytes."""
    with fitz.open(pdf_path) as src:
        out = fitz.open()
        out.insert_pdf(src, from_page=page_no - 1, to_page=page_no - 1)
        return out.tobytes()


def extract_tables_from_page(
    pdf_path: str,
    page_no: int,
    client: OpenAI,
    instructor_client: instructor.Instructor,
    model: str = "gpt-4o-mini",
) -> tuple[int, PageTables]:
    """Extract tables from a single 1-based page. Returns (page_no, PageTables)."""
    pdf_bytes = _slice_page(pdf_path, page_no)
    uploaded = client.files.create(
        file=("page.pdf", io.BytesIO(pdf_bytes), "application/pdf"),
        purpose="assistants",
    )
    try:
        result: PageTables = instructor_client.chat.completions.create(
            model=model,
            response_model=PageTables,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "file", "file": {"file_id": uploaded.id}},
                ],
            }],
        )
    finally:
        client.files.delete(uploaded.id)
    return page_no, result


def extract_all_table_pages(
    pdf_path: str,
    page_nos: list[int],
    client: OpenAI,
    model: str = "gpt-4o-mini",
    max_workers: int = 8,
) -> dict[int, PageTables]:
    """Extract tables from multiple pages in parallel.

    Returns dict mapping 1-based page_no → PageTables.
    Pages are sent simultaneously — latency is the slowest single page, not the sum.
    """
    instructor_client = instructor.from_openai(client)
    results: dict[int, PageTables] = {}

    failed_pages = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(page_nos))) as executor:
        futures = {
            executor.submit(extract_tables_from_page, pdf_path, page_no, client, instructor_client, model): page_no
            for page_no in page_nos
        }
        for future in as_completed(futures):
            page_no = futures[future]
            try:
                _, page_tables = future.result()
                results[page_no] = page_tables
                logger.info("page %d: extracted %d table(s)", page_no, len(page_tables.tables))
            except Exception as e:
                logger.warning("page %d: extraction failed (%s: %s) — original pymupdf text will be kept", page_no, type(e).__name__, e)
                failed_pages.append(page_no)

    if failed_pages:
        logger.warning("table extraction failed for %d page(s): %s", len(failed_pages), failed_pages)

    return results
