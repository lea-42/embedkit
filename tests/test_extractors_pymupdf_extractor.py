from unittest.mock import patch

import pytest

from unittest.mock import MagicMock

from docvec.extractors.models import PageTables, Table, TableCell
from docvec.extractors.pymupdf_extractor import PyMuPDFExtractor, find_table_pages, find_picture_table_pages, replace_picture_tables

SAMPLE_MD = """## Introduction

Some intro text.

-----

## Section Two

Second page content.
"""


@patch("docvec.extractors.pymupdf_extractor.pymupdf4llm.to_markdown", return_value=SAMPLE_MD)
def test_extract_returns_chunks(mock_md: object) -> None:
    chunks = PyMuPDFExtractor().extract("any.pdf")
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all("text" in c for c in chunks)


@patch("docvec.extractors.pymupdf_extractor.pymupdf4llm.to_markdown", return_value=SAMPLE_MD)
def test_chunks_have_required_keys(mock_md: object) -> None:
    chunk = PyMuPDFExtractor().extract("any.pdf")[0]
    assert "page_number" in chunk
    assert "chunk_number" in chunk
    assert "section_breadcrumbs" in chunk
    assert "text" in chunk


@patch("docvec.extractors.pymupdf_extractor.pymupdf4llm.to_markdown", return_value=SAMPLE_MD)
def test_max_pages_passed_to_pymupdf(mock_md: object) -> None:
    PyMuPDFExtractor(max_pages=5).extract("any.pdf")
    mock_md.assert_called_once()
    kwargs = mock_md.call_args.kwargs
    assert kwargs["pages"] == list(range(5))


@patch("docvec.extractors.pymupdf_extractor.pymupdf4llm.to_markdown", return_value=SAMPLE_MD)
def test_no_max_pages_omits_pages_kwarg(mock_md: object) -> None:
    PyMuPDFExtractor().extract("any.pdf")
    kwargs = mock_md.call_args.kwargs
    assert "pages" not in kwargs


@patch("docvec.extractors.pymupdf_extractor.pymupdf4llm.to_markdown", return_value="## A\n\n- n Item one\n- n Item two\n")
def test_bullet_n_stripped(mock_md: object) -> None:
    chunks = PyMuPDFExtractor().extract("any.pdf")
    text = chunks[0]["text"]
    assert "- n " not in text
    assert "Item one" in text


# --- find_table_pages ---

PICTURE_TABLE_MD = """\
Some prose.
--- end of page.page_number=2 ---

**----- Start of picture text -----**
Col A Col B
--- end of page.page_number=3 ---

More prose.
--- end of page.page_number=4 ---
"""

MARKDOWN_TABLE_MD = """\
Some prose.
--- end of page.page_number=1 ---

| Header A | Header B |
|---|---|
| cell 1 | cell 2 |
--- end of page.page_number=2 ---
"""

MIXED_TABLE_MD = """\
prose
**----- Start of picture text -----**
--- end of page.page_number=1 ---

| col |
|---|
--- end of page.page_number=2 ---

no table here
--- end of page.page_number=3 ---
"""


def test_find_table_pages_picture_marker() -> None:
    assert find_table_pages(PICTURE_TABLE_MD) == [3]


def test_find_table_pages_markdown_table() -> None:
    assert find_table_pages(MARKDOWN_TABLE_MD) == [2]


def test_find_table_pages_mixed() -> None:
    assert find_table_pages(MIXED_TABLE_MD) == [1, 2]


def test_find_table_pages_no_tables() -> None:
    md = "Some prose.\n--- end of page.page_number=1 ---\nMore prose.\n--- end of page.page_number=2 ---\n"
    assert find_table_pages(md) == []


def test_find_picture_table_pages_excludes_markdown_tables() -> None:
    # Markdown tables (|...|) are already correctly extracted by pymupdf4llm — not sent to OpenAI.
    assert find_picture_table_pages(MARKDOWN_TABLE_MD) == []


def test_find_picture_table_pages_detects_picture_marker() -> None:
    assert find_picture_table_pages(PICTURE_TABLE_MD) == [3]


def test_find_picture_table_pages_conduire() -> None:
    # Only garbled picture-text pages — excludes pages 3, 4 (TOC as markdown table)
    # and pages 11, 33 (already clean markdown tables).
    expected = [6, 7, 9, 12, 17, 21, 30, 31, 32, 48, 49]
    assert find_picture_table_pages(_conduire_md()) == expected


CONDUIRE_PDF = "tests/data/conditions-generales----conduire-1.pdf"


def _conduire_md() -> str:
    return PyMuPDFExtractor().extract_raw(CONDUIRE_PDF)


# --- replace_picture_tables ---

PICTURE_BLOCK_MD = """\
Some prose before.
**==> picture [100 x 50] intentionally omitted <==**

**----- Start of picture text -----**<br>
Col A Col B<br>
Val 1 Val 2<br>
**----- End of picture text -----**<br>

Some prose after.
--- end of page.page_number=5 ---

Next page content.
--- end of page.page_number=6 ---
"""

def _mock_page_tables(caption: str, rows: list[list[str]]) -> PageTables:
    """Build a PageTables from a 2D list of strings — row 0 is header."""
    cells = [
        TableCell(row=r, col=c, text=text)
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    return PageTables(tables=[Table(caption=caption, cells=cells)])


_PAGE_5_TABLES = _mock_page_tables("Test table", [["Col A", "Col B"], ["Val 1", "Val 2"]])


def test_replace_picture_tables_replaces_block() -> None:
    result = replace_picture_tables(PICTURE_BLOCK_MD, {5: _PAGE_5_TABLES})
    assert "Start of picture text" not in result
    assert "End of picture text" not in result
    assert "| Col A | Col B |" in result
    assert "| Val 1 | Val 2 |" in result


def test_replace_picture_tables_keeps_surrounding_prose() -> None:
    result = replace_picture_tables(PICTURE_BLOCK_MD, {5: _PAGE_5_TABLES})
    assert "Some prose before." in result
    assert "Some prose after." in result


def test_replace_picture_tables_untouched_page_unchanged() -> None:
    result = replace_picture_tables(PICTURE_BLOCK_MD, {5: _PAGE_5_TABLES})
    assert "Next page content." in result


def test_replace_picture_tables_no_results_unchanged() -> None:
    result = replace_picture_tables(PICTURE_BLOCK_MD, {})
    assert result == PICTURE_BLOCK_MD


def test_find_table_pages_conduire_full_doc() -> None:
    # Ground truth verified manually against the PDF (conditions-generales----conduire-1.pdf).
    #
    # Pages 3-4: table of contents, rendered as markdown table rows — correctly detected,
    # not a concern for OpenAI extraction but harmless to include.
    #
    # Pages 11 and 33: pymupdf4llm already renders these as proper markdown tables (|...|)
    # so they are detected correctly without any special handling.
    #
    # Page 50: contains a flowchart rendered as a dense wall of text with no formatting —
    # neither a picture-text block nor a markdown table. Known limitation, acceptable miss.
    expected = [3, 4, 6, 7, 9, 11, 12, 17, 21, 30, 31, 32, 33, 48, 49]
    assert find_table_pages(_conduire_md()) == expected
