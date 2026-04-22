from docvec.extractors.converters import json_to_markdown
from docvec.extractors.models import (
    DocumentExtractionBase,
    ImageRefBase,
    SectionBase,
    TableBase,
    TableCell,
)


def _doc(*sections: SectionBase) -> DocumentExtractionBase:
    return DocumentExtractionBase(sections=list(sections))


def _section(heading: str, level: int, page_start: int = 1, body: str = "") -> SectionBase:
    return SectionBase(heading=heading, level=level, page_start=page_start, body=body)


def _table(caption: str | None, rows: list[list[str]]) -> TableBase:
    cells = [TableCell(row=r, col=c, text=text) for r, row in enumerate(rows) for c, text in enumerate(row)]
    return TableBase(caption=caption, cells=cells)


def test_empty_document() -> None:
    assert json_to_markdown(_doc()) == ""


def test_single_section_heading_and_body() -> None:
    md = json_to_markdown(_doc(_section("Introduction", 1, body="Some intro text.")))
    assert "# Introduction" in md
    assert "Some intro text." in md


def test_heading_level_maps_to_hashes() -> None:
    md = json_to_markdown(_doc(
        _section("Top", 1),
        _section("Sub", 2),
        _section("SubSub", 3),
    ))
    assert "# Top" in md
    assert "## Sub" in md
    assert "### SubSub" in md


def test_table_rendered_as_markdown_table() -> None:
    table = _table("Coverage", [["Item", "Limit"], ["Fire", "10k"]])
    section = SectionBase(heading="Coverage", level=1, page_start=1, tables=[table])
    md = json_to_markdown(_doc(section))
    assert "| Item | Limit |" in md
    assert "| Fire | 10k |" in md


def test_image_renders_description_not_placeholder() -> None:
    image = ImageRefBase(page_start=1, caption="Figure 1", description="A bar chart.")
    section = SectionBase(heading="Charts", level=1, page_start=1, images=[image])
    md = json_to_markdown(_doc(section))
    assert "![" not in md
    assert "A bar chart." in md


def test_image_falls_back_to_caption() -> None:
    image = ImageRefBase(page_start=1, caption="Figure 1")
    section = SectionBase(heading="Charts", level=1, page_start=1, images=[image])
    md = json_to_markdown(_doc(section))
    assert "Figure 1" in md


def test_page_separator_between_pages() -> None:
    md = json_to_markdown(_doc(
        _section("Page One", 1, page_start=1, body="Text one."),
        _section("Page Two", 1, page_start=2, body="Text two."),
    ))
    assert "---" in md
    assert md.index("Text one.") < md.index("Text two.")


def test_no_separator_for_same_page() -> None:
    md = json_to_markdown(_doc(
        _section("A", 1, page_start=1, body="First."),
        _section("B", 2, page_start=1, body="Second."),
    ))
    assert "---" not in md
