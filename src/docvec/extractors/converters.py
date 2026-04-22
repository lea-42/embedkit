from docvec.extractors.models import (
    DocumentExtractionBase,
    ImageRefBase,
    SectionBase,
    TableBase,
)


def _table_to_markdown(table: TableBase) -> str:
    parts: list[str] = []
    if table.caption:
        parts.append(f"**{table.caption}**")
    if not table.cells:
        return "\n".join(parts)

    max_row = max(c.row for c in table.cells)
    max_col = max(c.col for c in table.cells)
    grid: list[list[str]] = [[""] * (max_col + 1) for _ in range(max_row + 1)]
    for cell in table.cells:
        grid[cell.row][cell.col] = cell.text

    for i, row in enumerate(grid):
        parts.append("| " + " | ".join(row) + " |")
        if i == 0:
            parts.append("| " + " | ".join("---" for _ in row) + " |")

    return "\n".join(parts)


def _image_to_markdown(image: ImageRefBase) -> str:
    text = image.description or image.caption
    return f"*{text}*" if text else ""


def _section_to_markdown(section: SectionBase) -> str:
    parts: list[str] = []
    hashes = "#" * section.level
    parts.append(f"{hashes} {section.heading}")
    if section.body:
        parts.append(section.body)
    for table in section.tables:
        parts.append(_table_to_markdown(table))
    for image in section.images:
        parts.append(_image_to_markdown(image))
    return "\n\n".join(parts)


def json_to_markdown(doc: DocumentExtractionBase) -> str:
    """Convert a DocumentExtractionBase to a markdown string.

    Sections are emitted in order. A page separator (---) is inserted
    whenever a section starts on a new page relative to the previous one.
    """
    parts: list[str] = []
    current_page: int | None = None

    for section in doc.sections:
        if current_page is not None and section.page_start != current_page:
            parts.append("---")
        current_page = section.page_start
        md = _section_to_markdown(section)
        if md.strip():
            parts.append(md)

    return "\n\n".join(parts)
