"""Scan a PDF and print which pages contain tables, using pymupdf4llm markers.

Usage:
    python scripts/find_table_pages.py <path/to/file.pdf>
"""
import sys

from docvec.extractors.pymupdf_extractor import PyMuPDFExtractor, find_table_pages

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "tests/data/conditions-generales--conduire-page-6.pdf"

md = PyMuPDFExtractor().extract_raw(pdf_path)
pages = find_table_pages(md)

print(f"PDF: {pdf_path}")
print(f"Table pages ({len(pages)}): {pages}")
