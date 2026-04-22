"""
Convert a saved OpenAI raw extraction JSON to markdown.

Usage:
    python scripts/raw_json_to_md.py tests/data/foo.pdf.openai.raw.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docvec.extractors.converters import json_to_markdown
from docvec.extractors.models import DocumentExtractionBase


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: raw_json_to_md.py <file.openai.raw.json>")
        sys.exit(1)

    raw_path = Path(sys.argv[1])
    out_path = raw_path.parent / raw_path.name.replace(".raw.json", ".raw.md")

    doc = DocumentExtractionBase.model_validate_json(raw_path.read_text())
    md = json_to_markdown(doc)
    out_path.write_text(md, encoding="utf-8")
    print(f"Markdown → {out_path}")


if __name__ == "__main__":
    main()
