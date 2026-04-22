from unittest.mock import MagicMock, patch

import pytest

from docvec.extractors.models import ScannedPDFError
from docvec.extractors.utils import detect_scanned


def _make_fitz_doc(texts: list[str]) -> MagicMock:
    """Build a mock fitz document whose pages return the given texts."""
    pages = []
    for text in texts:
        page = MagicMock()
        page.get_text.return_value = text
        pages.append(page)
    doc = MagicMock()
    doc.__iter__ = MagicMock(return_value=iter(pages))
    doc.__len__ = MagicMock(return_value=len(pages))
    doc.load_page.side_effect = lambda i: pages[i]
    return doc


@patch("docvec.extractors.utils.fitz.open")
def test_digital_pdf_passes(mock_open: MagicMock) -> None:
    page_text = "word " * 100  # 500 chars per page, clearly digital
    mock_open.return_value.__enter__ = lambda s: _make_fitz_doc([page_text, page_text, page_text])
    mock_open.return_value.__exit__ = MagicMock(return_value=False)
    detect_scanned("any.pdf")  # should not raise


@patch("docvec.extractors.utils.fitz.open")
def test_scanned_pdf_raises(mock_open: MagicMock) -> None:
    mock_open.return_value.__enter__ = lambda s: _make_fitz_doc(["", "", ""])
    mock_open.return_value.__exit__ = MagicMock(return_value=False)
    with pytest.raises(ScannedPDFError):
        detect_scanned("scanned.pdf")


@patch("docvec.extractors.utils.fitz.open")
def test_single_page_pdf_with_text_passes(mock_open: MagicMock) -> None:
    mock_open.return_value.__enter__ = lambda s: _make_fitz_doc(["word " * 100])  # 500 chars
    mock_open.return_value.__exit__ = MagicMock(return_value=False)
    detect_scanned("one_page.pdf")


@patch("docvec.extractors.utils.fitz.open")
def test_checks_at_most_three_pages(mock_open: MagicMock) -> None:
    """Only first 3 pages are sampled even for long docs."""
    texts = ["x" * 200] * 20  # 20 pages with plenty of text
    mock_open.return_value.__enter__ = lambda s: _make_fitz_doc(texts)
    mock_open.return_value.__exit__ = MagicMock(return_value=False)
    detect_scanned("long.pdf")


@patch("docvec.extractors.utils.fitz.open")
def test_sparse_text_raises(mock_open: MagicMock) -> None:
    """Very little text across sampled pages → scanned."""
    mock_open.return_value.__enter__ = lambda s: _make_fitz_doc(["hi", "", ""])
    mock_open.return_value.__exit__ = MagicMock(return_value=False)
    with pytest.raises(ScannedPDFError):
        detect_scanned("sparse.pdf")
