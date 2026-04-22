"""Test parallel table extraction on picture-text pages.

Usage:
    python scripts/test_table_extraction_prompt.py
"""
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from docvec.extractors.pymupdf_extractor import PyMuPDFExtractor, find_picture_table_pages
from docvec.extractors.table_extractor import PageTables, extract_all_table_pages

load_dotenv()

PDF_PATH = "tests/data/conditions-generales----conduire-1.pdf"
TEST_PAGES = [6, 7, 9]  # a few picture-text pages to test in parallel


def print_tables(page_no: int, result: PageTables) -> None:
    print(f"\n=== Page {page_no} — {len(result.tables)} table(s) ===")
    for t_idx, table in enumerate(result.tables):
        print(f"\n  Table {t_idx + 1}  caption={table.caption!r}")
        if not table.cells:
            print("  (no cells)")
            continue
        max_row = max(c.row for c in table.cells)
        max_col = max(c.col for c in table.cells)
        grid = [[""] * (max_col + 1) for _ in range(max_row + 1)]
        for c in table.cells:
            grid[c.row][c.col] = c.text
        col_widths = [max(len(grid[r][c]) for r in range(max_row + 1)) for c in range(max_col + 1)]
        for r_idx, row in enumerate(grid):
            line = " | ".join(cell.ljust(col_widths[c]) for c, cell in enumerate(row))
            print(f"  {line}")
            if r_idx == 0:
                print("  " + "-+-".join("-" * w for w in col_widths))


def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"Extracting {len(TEST_PAGES)} pages in parallel: {TEST_PAGES}\n")
    t0 = time.perf_counter()
    results = extract_all_table_pages(PDF_PATH, TEST_PAGES, client)
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.2f}s (parallel)\n")

    for page_no in sorted(results):
        print_tables(page_no, results[page_no])
        out_path = f"tests/data/conditions-generales----conduire-1.pdf.page_{page_no}.tables.json"
        with open(out_path, "w") as f:
            json.dump(results[page_no].model_dump(), f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
