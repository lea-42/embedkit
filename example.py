"""Example runner — shows the full pipeline and saves output files next to the PDF."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from docvec.logging_config import setup_logging
from docvec.extractors.pymupdf_extractor import PyMuPDFExtractor
from docvec.embedder import DEFAULT_MODEL, load_model, embed_chunks

setup_logging()

if len(sys.argv) < 2:
    print("Usage: python example.py <pdf_path>")
    sys.exit(1)

pdf_path = Path(sys.argv[1])

# --- extract and chunk ---
chunks = PyMuPDFExtractor().extract(str(pdf_path))
print(f"Extracted {len(chunks)} chunks")

# save markdown and chunks next to the PDF
chunks_path = pdf_path.parent / (pdf_path.name + "_chunks.json")
chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved {chunks_path.name}")

print(f"\nFirst 3 chunks:")
for c in chunks[:3]:
    print(json.dumps(c, ensure_ascii=False, indent=2))

# --- vectorize ---
print(f"\nLoading model {DEFAULT_MODEL.model_name}...")
asyncio.run(load_model(DEFAULT_MODEL))
pairs = list(embed_chunks(chunks, config=DEFAULT_MODEL))
print(f"Embedded {len(pairs)} chunks, dimension: {pairs[0][1].shape[0]}")
