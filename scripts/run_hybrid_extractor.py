"""Run HybridExtractor on a PDF and save all outputs.

Usage:
    python scripts/run_hybrid_extractor.py [path/to/file.pdf]
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from docvec.extractors.hybrid_extractor import HybridExtractor

load_dotenv()

PDF_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/data/conditions-generales----conduire-1.pdf")
MD_OUT = Path(str(PDF_PATH) + ".hybrid.raw.md")
CHUNKS_OUT = Path(str(PDF_PATH) + ".hybrid.json")

print(f"PDF: {PDF_PATH}")
print()

t0 = time.perf_counter()
extractor = HybridExtractor(api_key=os.environ["OPENAI_API_KEY"])
md = extractor.extract_raw(str(PDF_PATH))
t_extract = time.perf_counter() - t0

from docvec.chunker import chunk_markdown
t1 = time.perf_counter()
chunks = chunk_markdown(md)
t_chunk = time.perf_counter() - t1

MD_OUT.write_text(md, encoding="utf-8")
CHUNKS_OUT.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))

print(f"\n--- timings ---")
print(f"  extraction : {t_extract:.2f}s")
print(f"  chunking   : {t_chunk:.2f}s")
print(f"  total      : {t_extract + t_chunk:.2f}s")
print(f"\n--- outputs ---")
print(f"  {MD_OUT}  ({len(md)} chars)")
print(f"  {CHUNKS_OUT}  ({len(chunks)} chunks)")
