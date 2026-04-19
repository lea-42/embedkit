"""
Live integration tests for OpenAIExtractor.
Requires a real OpenAI API key and a PDF at tests/data/aig-car-policy-09-2021.pdf.

Run with:
    pytest tests/test_extractors_openai_live.py --live
"""
import json
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_PDF = DATA_DIR / "aig-car-policy-09-2021.pdf"

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def skip_unless_live(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--live"):
        pytest.skip("live test — run with --live")


@pytest.fixture(autouse=True)
def require_sample_pdf() -> None:
    if not SAMPLE_PDF.exists():
        pytest.skip(f"sample PDF not found at {SAMPLE_PDF} — add a PDF to run live tests")


def _save_and_print(chunks: list[dict], suffix: str, pages: int, elapsed: float) -> None:
    output_path = DATA_DIR / (SAMPLE_PDF.name + f".{suffix}.json")
    output_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    per_page = elapsed / pages if pages else 0
    print(f"\n[{suffix}] {len(chunks)} chunks | {pages} pages | {elapsed:.1f}s total | {per_page:.2f}s/page")
    print(f"Saved to {output_path}", flush=True)
    for i, chunk in enumerate(chunks):
        print(f"\n--- chunk {i} (page {chunk.get('page_number')}) ---", flush=True)
        print(chunk["text"], flush=True)


def test_openai_extractor_batching_live() -> None:
    """Extract first 11 pages with batch_size=5 — expect 3 batches, chunks returned."""
    from docvec.extractors.openai_extractor import OpenAIExtractor

    pages = 11
    extractor = OpenAIExtractor(batch_size=5, max_pages=pages)

    t0 = time.time()
    doc = extractor.extract_raw(str(SAMPLE_PDF))
    elapsed = time.time() - t0

    # Save raw extraction so chunking can be iterated without re-calling OpenAI
    raw_path = DATA_DIR / (SAMPLE_PDF.name + ".openai.raw.json")
    raw_path.write_text(doc.model_dump_json(indent=2))
    print(f"\n[openai] {len(doc.sections)} sections extracted", flush=True)
    for s in doc.sections:
        print(
            f"  page {s.page_start} level={s.level}: {s.heading!r} body={len(s.body)}chars "
            f"tables={len(s.tables)} images={len(s.images)} subsections={len(s.subsections)}",
            flush=True,
        )

    from docvec.extractors.converters import json_to_markdown
    from docvec.chunker import chunk_markdown
    md = json_to_markdown(doc)
    print(f"\n[openai] markdown length: {len(md)} chars", flush=True)
    print(f"[openai] markdown preview:\n{md[:500]}", flush=True)
    chunks = chunk_markdown(md)
    print(f"[openai] {len(chunks)} chunks", flush=True)

    assert isinstance(chunks, list), "extract() must return a list"
    assert len(chunks) > 0, "expected at least one chunk from 11 pages"
    for chunk in chunks:
        assert "text" in chunk, f"chunk missing 'text' key: {chunk}"
        assert isinstance(chunk["text"], str)
        assert len(chunk["text"]) > 0

    _save_and_print(chunks, "openai", pages, elapsed)


def test_pymupdf_extractor_live() -> None:
    """Extract same pages with PyMuPDFExtractor for comparison."""
    from docvec.extractors.pymupdf_extractor import PyMuPDFExtractor

    pages = 11
    t0 = time.time()
    chunks = PyMuPDFExtractor().extract(str(SAMPLE_PDF))
    elapsed = time.time() - t0

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    _save_and_print(chunks, "pymupdf", pages, elapsed)
