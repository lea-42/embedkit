from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docvec.extractors.models import (
    BatchExtractionBase,
    ScannedPDFError,
    SectionBase,
)
from docvec.extractors.openai_extractor import OpenAIExtractor


SAMPLE_BATCH = BatchExtractionBase(
    sections=[SectionBase(heading="Introduction", level=1, page_start=1, body="Hello world.")],
    open_sections=["Introduction"],
)

EMPTY_BATCH = BatchExtractionBase(sections=[], open_sections=[])

FAKE_PDF_BYTES = b"%PDF-1.4 fake"


def _make_extractor() -> OpenAIExtractor:
    return OpenAIExtractor(api_key="test-key", model="gpt-4o", batch_size=10)


def _setup_instructor_mock(mock_instructor: MagicMock, mock_openai: MagicMock, batch_return: BatchExtractionBase = SAMPLE_BATCH) -> tuple[MagicMock, MagicMock]:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.files.create.return_value = MagicMock(id="file-123")

    mock_instructor_client = MagicMock()
    mock_instructor.from_openai.return_value = mock_instructor_client
    mock_instructor_client.chat.completions.create.return_value = batch_return

    return mock_client, mock_instructor_client


@patch("docvec.extractors.openai_extractor._slice_pdf", return_value=FAKE_PDF_BYTES)
@patch("docvec.extractors.openai_extractor.get_page_count", return_value=1)
@patch("docvec.extractors.openai_extractor.detect_scanned")
@patch("docvec.extractors.openai_extractor.instructor")
@patch("docvec.extractors.openai_extractor.OpenAI")
def test_extract_returns_chunks(mock_openai: MagicMock, mock_instructor: MagicMock, mock_detect: MagicMock, mock_page_count: MagicMock, mock_slice: MagicMock, tmp_path: Path) -> None:
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(FAKE_PDF_BYTES)
    _setup_instructor_mock(mock_instructor, mock_openai)

    chunks = _make_extractor().extract(str(pdf))

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all("text" in c for c in chunks)


@patch("docvec.extractors.openai_extractor._slice_pdf", return_value=FAKE_PDF_BYTES)
@patch("docvec.extractors.openai_extractor.get_page_count", return_value=1)
@patch("docvec.extractors.openai_extractor.detect_scanned")
@patch("docvec.extractors.openai_extractor.instructor")
@patch("docvec.extractors.openai_extractor.OpenAI")
def test_uploaded_file_deleted_on_success(mock_openai: MagicMock, mock_instructor: MagicMock, mock_detect: MagicMock, mock_page_count: MagicMock, mock_slice: MagicMock, tmp_path: Path) -> None:
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(FAKE_PDF_BYTES)
    mock_client, _ = _setup_instructor_mock(mock_instructor, mock_openai)
    mock_client.files.create.return_value = MagicMock(id="file-abc")

    _make_extractor().extract(str(pdf))

    mock_client.files.delete.assert_called_once_with("file-abc")


@patch("docvec.extractors.openai_extractor._slice_pdf", return_value=FAKE_PDF_BYTES)
@patch("docvec.extractors.openai_extractor.get_page_count", return_value=1)
@patch("docvec.extractors.openai_extractor.detect_scanned")
@patch("docvec.extractors.openai_extractor.instructor")
@patch("docvec.extractors.openai_extractor.OpenAI")
def test_uploaded_file_deleted_on_failure(mock_openai: MagicMock, mock_instructor: MagicMock, mock_detect: MagicMock, mock_page_count: MagicMock, mock_slice: MagicMock, tmp_path: Path) -> None:
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(FAKE_PDF_BYTES)

    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.files.create.return_value = MagicMock(id="file-xyz")

    mock_instructor_client = MagicMock()
    mock_instructor.from_openai.return_value = mock_instructor_client
    mock_instructor_client.chat.completions.create.side_effect = RuntimeError("API error")

    with pytest.raises(RuntimeError):
        _make_extractor().extract(str(pdf))

    mock_client.files.delete.assert_called_once_with("file-xyz")


@patch("docvec.extractors.openai_extractor.detect_scanned", side_effect=ScannedPDFError("scanned"))
@patch("docvec.extractors.openai_extractor.OpenAI")
def test_scanned_pdf_raises(mock_openai: MagicMock, mock_detect: MagicMock, tmp_path: Path) -> None:
    pdf = tmp_path / "scanned.pdf"
    pdf.write_bytes(FAKE_PDF_BYTES)

    with pytest.raises(ScannedPDFError):
        _make_extractor().extract(str(pdf))


@patch("docvec.extractors.openai_extractor._slice_pdf", return_value=FAKE_PDF_BYTES)
@patch("docvec.extractors.openai_extractor.get_page_count", return_value=5)
@patch("docvec.extractors.openai_extractor.detect_scanned")
@patch("docvec.extractors.openai_extractor.instructor")
@patch("docvec.extractors.openai_extractor.OpenAI")
def test_batching_splits_pages(mock_openai: MagicMock, mock_instructor: MagicMock, mock_detect: MagicMock, mock_page_count: MagicMock, mock_slice: MagicMock, tmp_path: Path) -> None:
    """With batch_size=2 and a 5-page doc, expect 3 API calls."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(FAKE_PDF_BYTES)
    _, mock_instructor_client = _setup_instructor_mock(mock_instructor, mock_openai, EMPTY_BATCH)

    extractor = OpenAIExtractor(api_key="test-key", model="gpt-4o", batch_size=2)
    extractor.extract(str(pdf))

    assert mock_instructor_client.chat.completions.create.call_count == 3  # ceil(5/2)


@patch("docvec.extractors.openai_extractor._slice_pdf", return_value=FAKE_PDF_BYTES)
@patch("docvec.extractors.openai_extractor.get_page_count", return_value=1)
@patch("docvec.extractors.openai_extractor.detect_scanned")
@patch("docvec.extractors.openai_extractor.instructor")
@patch("docvec.extractors.openai_extractor.OpenAI")
def test_open_sections_passed_to_next_batch(mock_openai: MagicMock, mock_instructor: MagicMock, mock_detect: MagicMock, mock_page_count: MagicMock, mock_slice: MagicMock, tmp_path: Path) -> None:
    """open_sections from batch N are passed as context to batch N+1 prompt."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(FAKE_PDF_BYTES)

    mock_page_count.return_value = 4

    batch1 = BatchExtractionBase(
        sections=[SectionBase(heading="Chapter 1", level=1, page_start=1, body="Intro.")],
        open_sections=["Chapter 1", "1.1 Sub"],
    )
    batch2 = BatchExtractionBase(sections=[], open_sections=[])

    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.files.create.return_value = MagicMock(id="file-x")

    mock_instructor_client = MagicMock()
    mock_instructor.from_openai.return_value = mock_instructor_client
    mock_instructor_client.chat.completions.create.side_effect = [batch1, batch2]

    extractor = OpenAIExtractor(api_key="test-key", model="gpt-4o", batch_size=2)
    doc = extractor.extract_raw(str(pdf))

    # Verify second call prompt contains the open_sections from batch1
    second_call_kwargs = mock_instructor_client.chat.completions.create.call_args_list[1].kwargs
    prompt_text = second_call_kwargs["messages"][0]["content"][0]["text"]
    assert "Chapter 1" in prompt_text
    assert "1.1 Sub" in prompt_text

    # Sections from both batches are merged
    assert len(doc.sections) == 1
    assert doc.sections[0].heading == "Chapter 1"
