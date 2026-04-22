import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import instructor
from openai import OpenAI

logger = logging.getLogger(__name__)

from docvec.chunker import chunk_markdown
from docvec.extractors.base import BaseExtractor
from docvec.extractors.converters import json_to_markdown
from docvec.extractors.models import (
    BatchExtractionBase,
    DocumentExtractionBase,
    SectionBase,
)
from docvec.extractors.utils import detect_scanned


def balanced_batches(page_count: int, min_size: int = 10, max_size: int = 15) -> list[tuple[int, int]]:
    """Split page_count into evenly-sized batches between min_size and max_size.

    Picks the number of batches that minimises size variance, then distributes
    remainder pages one-at-a-time to the first batches so sizes differ by at most 1.
    Returns list of (start, end) page ranges (0-based, end exclusive).
    """
    target_batches = max(1, round(page_count / ((min_size + max_size) / 2)))
    base, remainder = divmod(page_count, target_batches)

    # Clamp: if base size falls outside [min, max], nudge target_batches
    if base < min_size and target_batches > 1:
        target_batches -= 1
        base, remainder = divmod(page_count, target_batches)
    elif base > max_size:
        target_batches += 1
        base, remainder = divmod(page_count, target_batches)

    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(target_batches):
        size = base + (1 if i < remainder else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def get_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF."""
    with fitz.open(pdf_path) as doc:
        return len(doc)


def _slice_pdf(pdf_path: str, start: int, end: int) -> bytes:
    """Extract pages [start, end) from a PDF and return as bytes."""
    with fitz.open(pdf_path) as src:
        out = fitz.open()
        out.insert_pdf(src, from_page=start, to_page=end - 1)
        return out.tobytes()


def _build_prompt(batch_start: int, batch_end: int, open_sections: list[str]) -> str:
    context = f"You are reading pages {batch_start + 1}–{batch_end} of a larger document."
    if open_sections:
        path = " > ".join(open_sections)
        context += f" The document was mid-way through sections: {path!r}. Continue extracting from that context."
    return (
        f"{context}\n\n"
        "Extract ALL content from these pages as a FLAT list of sections. Rules:\n"
        "- Every heading becomes its own section entry — do NOT nest subsections inside a parent\n"
        "- Each section has: heading, level, page_start, and body\n"
        "- level reflects the heading's depth in the document hierarchy (1=top, 2=sub, 3=sub-sub, etc.). "
        "Infer it from: visual size (larger = higher), numbering (2=level 1, 2.1=level 2, 2.1.1=level 3), "
        "and indentation. Different headings on the same page will often have different levels.\n"
        "- body is all text under that heading up to the next heading of equal or higher level\n"
        "- page_start is the 1-based page number the heading appears on\n"
        "- TABLES: whenever you see a table, you MUST:\n"
        "  1. Put it in the `tables` field as structured cells (row 0-based, col 0-based, text per cell)\n"
        "  2. Leave the table content OUT of body entirely — body contains only the prose before and after the table\n"
        "  3. Do NOT add an image entry describing the table\n"
        "  Row 0 is the header row. Include every cell, use empty string for empty cells.\n"
        "  Example: 'Name | Age / Alice | 30' → cells: [{row:0,col:0,text:'Name'},{row:0,col:1,text:'Age'},{row:1,col:0,text:'Alice'},{row:1,col:1,text:'30'}]\n"
        "- IMAGES: only populate `images` for actual images/figures/diagrams — never for tables\n"
        "- If a section spans multiple pages, use a single entry with the page_start where the heading appears\n"
        "- After listing all sections, set open_sections to the full heading path of the deepest "
        "open section at the end of these pages (e.g. ['2 Benefits', '2.1 Travel Exclusions'])\n"
        "- Render all bullet points and list items using '- ' (hyphen-space), regardless of how they appear in the PDF\n"
        "- Do not summarize or skip content. Preserve all text exactly."
    )


class OpenAIExtractor(BaseExtractor):
    """Slow, accurate extractor using OpenAI + instructor. Costs API tokens.

    Slices the PDF into batches of `batch_size` pages, uploads each slice,
    extracts structured JSON via instructor, then converts to markdown chunks.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_pages: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.max_pages = max_pages

    def extract_raw(self, pdf_path: str) -> DocumentExtractionBase:
        """Call OpenAI and return the structured extraction result.

        Use this to cache the raw extraction and iterate on chunking separately.
        """
        detect_scanned(pdf_path)

        client = OpenAI(api_key=self.api_key)
        instructor_client = instructor.from_openai(client)
        page_count = get_page_count(pdf_path)
        if self.max_pages is not None:
            page_count = min(page_count, self.max_pages)
        batches = balanced_batches(page_count)
        logger.info("[plan] %d pages → %d batches: %s", page_count, len(batches), [e - s for s, e in batches])

        all_sections: list[SectionBase] = []
        open_sections: list[str] = []

        t_total = time.perf_counter()

        pdf_path_obj = Path(pdf_path)
        cache_dir = pdf_path_obj.parent / "__docvec_cache__"
        cache_dir.mkdir(exist_ok=True)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            # 1-based inclusive page range in filename, e.g. pages_1_13
            cache_path = cache_dir / f"{pdf_path_obj.name}.pages_{batch_start + 1}_{batch_end}.batch.json"

            if cache_path.exists():
                envelope = json.loads(cache_path.read_text())
                cached = BatchExtractionBase.model_validate(envelope["batch"])
                logger.info("[batch %d] cache hit: %s  model=%s  ts=%s  sections=%d", batch_idx, cache_path.name, envelope.get("model"), envelope.get("timestamp"), len(cached.sections))
                all_sections.extend(cached.sections)
                open_sections = cached.open_sections
                continue

            t0 = time.perf_counter()
            pdf_slice = _slice_pdf(pdf_path, batch_start, batch_end)
            t_slice = time.perf_counter() - t0
            logger.info("[batch %d] slice=%.2fs  (%db)", batch_idx, t_slice, len(pdf_slice))

            file_id: str | None = None

            try:
                t0 = time.perf_counter()
                uploaded = client.files.create(
                    file=("batch.pdf", io.BytesIO(pdf_slice), "application/pdf"),
                    purpose="assistants",
                )
                file_id = uploaded.id
                t_upload = time.perf_counter() - t0
                logger.info("[batch %d] upload=%.2fs  file_id=%s", batch_idx, t_upload, file_id)

                t0 = time.perf_counter()
                batch: BatchExtractionBase = instructor_client.chat.completions.create(
                    model=self.model,
                    response_model=BatchExtractionBase,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": _build_prompt(batch_start, batch_end, open_sections),
                                },
                                {
                                    "type": "file",
                                    "file": {"file_id": file_id},
                                },
                            ],
                        }
                    ],
                )
                t_extract = time.perf_counter() - t0
                logger.info("[batch %d] extract=%.2fs  sections=%d", batch_idx, t_extract, len(batch.sections))

                envelope = {
                    "model": self.model,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "extract_seconds": round(t_extract, 2),
                    "pages": f"{batch_start + 1}-{batch_end}",
                    "batch": batch.model_dump(),
                }
                cache_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False))
                logger.info("[batch %d] cached → %s", batch_idx, cache_path.name)

                all_sections.extend(batch.sections)
                open_sections = batch.open_sections

            finally:
                if file_id is not None:
                    try:
                        t0 = time.perf_counter()
                        client.files.delete(file_id)
                        t_delete = time.perf_counter() - t0
                        logger.info("[batch %d] delete=%.2fs", batch_idx, t_delete)
                    except Exception as e:
                        logger.warning("[batch %d] cleanup failed: %s: %s", batch_idx, type(e).__name__, e)

        logger.info("total=%.2fs", time.perf_counter() - t_total)

        return DocumentExtractionBase(sections=all_sections)

    def extract(self, pdf_path: str) -> list[dict]:
        doc = self.extract_raw(pdf_path)
        md = json_to_markdown(doc)
        return chunk_markdown(md)
