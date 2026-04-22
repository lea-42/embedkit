"""Extract a PDF with the OpenAI extractor and save raw JSON next to the file.

Usage:
    python scripts/extract_page6_openai.py [path/to/file.pdf]

Defaults to tests/data/conditions-generales--conduire-page-6.pdf if no arg given.
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from docvec.chunker import chunk_markdown
from docvec.extractors.converters import json_to_markdown
from docvec.extractors.openai_extractor import OpenAIExtractor, balanced_batches, get_page_count

load_dotenv()

PDF_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/data/conditions-generales--conduire-page-6.pdf")
RAW_OUT = Path(str(PDF_PATH) + ".openai.raw.json")
MD_OUT = Path(str(PDF_PATH) + ".openai.raw.md")
CHUNKS_OUT = Path(str(PDF_PATH) + ".openai.json")
TIMING_OUT = Path(str(PDF_PATH) + ".timing.txt")

# Preview batch plan and ask for confirmation
page_count = get_page_count(str(PDF_PATH))
batches = balanced_batches(page_count)
sizes = [e - s for s, e in batches]
print(f"PDF:     {PDF_PATH}")
print(f"Pages:   {page_count}")
print(f"Batches: {len(batches)}  sizes={sizes}")
print(f"Output:  {RAW_OUT.parent}/")
print()
confirm = input("Proceed? [y/N] ").strip().lower()
if confirm != "y":
    print("Aborted.")
    sys.exit(0)
print()

extractor = OpenAIExtractor(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",
)

# Intercept prints from the extractor to capture timing lines
timing_lines: list[str] = []
_orig_print = print

def _tee_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if any(k in msg for k in ("slice=", "upload=", "extract=", "delete=", "total=", "[plan]")):
        timing_lines.append(msg)
    _orig_print(*args, **kwargs)

import builtins
builtins.print = _tee_print

t_wall = time.perf_counter()
print(f"Extracting {PDF_PATH} ...\n")
doc = extractor.extract_raw(str(PDF_PATH))
t_wall = time.perf_counter() - t_wall

builtins.print = _orig_print

RAW_OUT.write_text(json.dumps(doc.model_dump(), indent=2, ensure_ascii=False))
print(f"Saved: {RAW_OUT}")

TIMING_OUT.write_text(f"wall_time={t_wall:.2f}s\n\n" + "\n".join(timing_lines) + "\n", encoding="utf-8")
print(f"Saved: {TIMING_OUT}")

md = json_to_markdown(doc)
MD_OUT.write_text(md, encoding="utf-8")
print(f"Saved: {MD_OUT}")

chunks = chunk_markdown(md)
CHUNKS_OUT.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
print(f"Saved: {CHUNKS_OUT}  ({len(chunks)} chunks)")

print("\n=== STRUCTURED JSON ===")
for section in doc.sections:
    print(f"  [L{section.level}] {section.heading!r}  (page {section.page_start})")
    if section.body:
        print(f"    body: {section.body[:120].replace(chr(10), ' ')!r}")
    for t in section.tables:
        print(f"    table: {len(t.cells)} cells  caption={t.caption!r}")

print("\n=== MARKDOWN OUTPUT ===")
print(md)
