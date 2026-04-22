"""Unit tests for HybridExtractor — LLM calls are mocked."""
from unittest.mock import MagicMock, patch

from docvec.extractors.hybrid_extractor import HybridExtractor
from docvec.extractors.models import PageTables, Table, TableCell

PYMUPDF_MD = """\
Some prose.
**----- Start of picture text -----**<br>
Garbled Col A Garbled Col B<br>
**----- End of picture text -----**<br>
More prose.
--- end of page.page_number=3 ---
"""

MOCK_TABLE = PageTables(tables=[
    Table(
        caption=None,
        cells=[
            TableCell(row=0, col=0, text="Col A"),
            TableCell(row=0, col=1, text="Col B"),
            TableCell(row=1, col=0, text="Val 1"),
            TableCell(row=1, col=1, text="Val 2"),
        ],
    )
])


@patch.object(HybridExtractor, "_extract_tables", return_value={3: MOCK_TABLE})
@patch.object(HybridExtractor, "_pymupdf_raw", return_value=PYMUPDF_MD)
def test_hybrid_replaces_picture_table(mock_pymupdf: MagicMock, mock_tables: MagicMock) -> None:
    result = HybridExtractor().extract_raw("any.pdf")
    assert "Start of picture text" not in result
    assert "| Col A | Col B |" in result
    assert "| Val 1 | Val 2 |" in result


@patch.object(HybridExtractor, "_extract_tables", return_value={3: MOCK_TABLE})
@patch.object(HybridExtractor, "_pymupdf_raw", return_value=PYMUPDF_MD)
def test_hybrid_keeps_prose(mock_pymupdf: MagicMock, mock_tables: MagicMock) -> None:
    result = HybridExtractor().extract_raw("any.pdf")
    assert "Some prose." in result
    assert "More prose." in result


NO_TABLE_MD = """\
Just prose, no tables.
--- end of page.page_number=1 ---
"""


@patch.object(HybridExtractor, "_extract_tables")
@patch.object(HybridExtractor, "_pymupdf_raw", return_value=NO_TABLE_MD)
def test_hybrid_skips_openai_when_no_tables(mock_pymupdf: MagicMock, mock_tables: MagicMock) -> None:
    HybridExtractor().extract_raw("any.pdf")
    mock_tables.assert_not_called()


@patch.object(HybridExtractor, "_extract_tables", return_value={3: MOCK_TABLE})
@patch.object(HybridExtractor, "_pymupdf_raw", return_value=PYMUPDF_MD)
def test_hybrid_extract_returns_chunks(mock_pymupdf: MagicMock, mock_tables: MagicMock) -> None:
    chunks = HybridExtractor().extract("any.pdf")
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all("text" in c for c in chunks)
