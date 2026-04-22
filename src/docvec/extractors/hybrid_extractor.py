"""Hybrid extractor: pymupdf4llm for text, OpenAI for picture-text tables.

Strategy:
  1. Extract full doc with pymupdf4llm (fast, free, good read order)
  2. Detect pages with garbled picture-text tables
  3. Send those pages to OpenAI in parallel (table-only prompt, fast)
  4. Replace garbled blocks in the markdown with clean extracted tables
  5. Chunk the patched markdown
"""
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

from docvec.chunker import chunk_markdown
from docvec.extractors.base import BaseExtractor
from docvec.extractors.pymupdf_extractor import (
    PyMuPDFExtractor,
    find_picture_table_pages,
    replace_picture_tables,
)
from docvec.extractors.table_extractor import extract_all_table_pages


class HybridExtractor(BaseExtractor):
    """Fast extraction using pymupdf4llm with OpenAI table correction for picture-text tables."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_workers: int = 8,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.max_workers = max_workers

    def _pymupdf_raw(self, pdf_path: str) -> str:
        return PyMuPDFExtractor().extract_raw(pdf_path)

    def _find_table_pages(self, md: str) -> list[int]:
        return find_picture_table_pages(md)

    def _extract_tables(self, pdf_path: str, page_nos: list[int]) -> dict:
        client = OpenAI(api_key=self.api_key)
        return extract_all_table_pages(pdf_path, page_nos, client, model=self.model, max_workers=self.max_workers)

    def _patch_markdown(self, md: str, table_results: dict) -> str:
        return replace_picture_tables(md, table_results)

    def extract_raw(self, pdf_path: str) -> str:
        """Return the patched markdown — pymupdf text with OpenAI tables inserted."""
        logger.info("extracting with pymupdf4llm")
        md = self._pymupdf_raw(pdf_path)

        table_pages = self._find_table_pages(md)
        logger.info("found %d picture-text table page(s): %s", len(table_pages), table_pages)

        if not table_pages:
            return md

        logger.info("extracting tables in parallel")
        table_results = self._extract_tables(pdf_path, table_pages)
        logger.info("received table results for %d/%d page(s)", len(table_results), len(table_pages))

        logger.info("patching markdown")
        return self._patch_markdown(md, table_results)

    def extract(self, pdf_path: str) -> list[dict]:
        md = self.extract_raw(pdf_path)
        return chunk_markdown(md)
