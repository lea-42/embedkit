from unittest.mock import patch

import pytest

from docvec.extractors.pymupdf_extractor import PyMuPDFExtractor

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
