"""
Re-chunk a saved OpenAI raw extraction without calling the API again.

Usage:
    python scripts/rechunk_openai.py tests/data/foo.pdf.openai.raw.json
    python scripts/rechunk_openai.py tests/data/foo.pdf.openai.raw.json --out tests/data/foo.pdf.openai.json
"""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docvec.chunker import chunk_markdown
from docvec.extractors.converters import json_to_markdown
from docvec.extractors.models import DocumentExtractionBase


def rechunk(raw_path: Path, out_path: Path) -> list[dict]:
    doc = DocumentExtractionBase.model_validate_json(raw_path.read_text())
    md = json_to_markdown(doc)
    chunks = chunk_markdown(md)
    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    print(f"{len(chunks)} chunks → {out_path}")
    for i, chunk in enumerate(chunks):
        print(f"\n--- chunk {i} (page {chunk.get('page_number')}, breadcrumb: {' > '.join(chunk.get('section_breadcrumbs', []))}) ---")
        print(chunk["text"])
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-chunk a saved OpenAI raw extraction.")
    parser.add_argument("raw", help="Path to .openai.raw.json file")
    parser.add_argument("--out", default=None, help="Output path (default: replaces .raw.json with .json)")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_path = Path(args.out) if args.out else Path(str(raw_path).replace(".raw.json", ".json"))
    rechunk(raw_path, out_path)


if __name__ == "__main__":
    main()
