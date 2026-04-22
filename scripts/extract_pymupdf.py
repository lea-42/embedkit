"""
Extract and chunk a PDF with PyMuPDFExtractor, save chunks and raw markdown.

Usage:
    python scripts/extract_pymupdf.py tests/data/foo.pdf
    python scripts/extract_pymupdf.py tests/data/foo.pdf --max-pages 11
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docvec.extractors.pymupdf_extractor import PyMuPDFExtractor
from docvec.chunker import chunk_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and chunk a PDF with PyMuPDFExtractor.")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--max-pages", type=int, default=None, help="Only extract first N pages")
    parser.add_argument("--out", default=None, help="Output path (default: <pdf>.pymupdf.json)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out) if args.out else pdf_path.parent / (pdf_path.name + ".pymupdf.json")
    raw_path = pdf_path.parent / (pdf_path.name + ".pymupdf.raw.md")

    extractor = PyMuPDFExtractor(max_pages=args.max_pages)
    md = extractor.extract_raw(str(pdf_path))

    raw_path.write_text(md, encoding="utf-8")
    print(f"Raw markdown → {raw_path}")

    chunks = chunk_markdown(md)
    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    print(f"{len(chunks)} chunks → {out_path}")


if __name__ == "__main__":
    main()
