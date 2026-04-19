from docvec.chunker import chunk_markdown

SIMPLE_MD = """# Travel Insurance

## What is Insured

### Checked Baggage

This is a paragraph about checked baggage.

Another paragraph about limits.

## What is Not Insured

Exclusions paragraph here.
"""

TWO_PAGE_MD = """# Section One

First page paragraph.

-----

# Section One

## Sub Section

Second page paragraph.
"""

TABLE_MD = """# Policy

## Coverage

| Col A | Col B |
|---|---|
| val 1 | val 2 |

Next paragraph after table.
"""

LIST_MD = """# Policy

## Benefits

- Item one
- Item two
- Item three

Next paragraph.
"""


def test_chunks_are_returned() -> None:
    chunks = chunk_markdown(SIMPLE_MD)
    assert len(chunks) > 0


def test_chunk_has_required_keys() -> None:
    chunk = chunk_markdown(SIMPLE_MD)[0]
    assert "page_number" in chunk
    assert "chunk_number" in chunk
    assert "section_breadcrumbs" in chunk
    assert "text" in chunk


def test_chunk_numbers_are_sequential() -> None:
    chunks = chunk_markdown(SIMPLE_MD)
    assert [c["chunk_number"] for c in chunks] == list(range(len(chunks)))


def test_breadcrumbs_track_current_heading() -> None:
    chunks = chunk_markdown(SIMPLE_MD)
    baggage_chunk = next(c for c in chunks if "checked baggage" in c["text"].lower())
    # Full hierarchy: h1 > h2 > h3
    assert baggage_chunk["section_breadcrumbs"] == ["Travel Insurance", "What is Insured", "Checked Baggage"]


def test_breadcrumbs_update_on_new_section() -> None:
    chunks = chunk_markdown(SIMPLE_MD)
    exclusions_chunk = next(c for c in chunks if "exclusions" in c["text"].lower())
    # h2 resets under h1, no h3
    assert exclusions_chunk["section_breadcrumbs"] == ["Travel Insurance", "What is Not Insured"]


def test_page_number_increments_at_separator() -> None:
    chunks = chunk_markdown(TWO_PAGE_MD)
    page_numbers = [c["page_number"] for c in chunks]
    assert page_numbers[0] == 1
    assert any(c["page_number"] == 2 for c in chunks)


def test_table_is_single_chunk() -> None:
    chunks = chunk_markdown(TABLE_MD)
    table_chunks = [c for c in chunks if "|" in c["text"]]
    assert len(table_chunks) == 1
    assert "val 1" in table_chunks[0]["text"]


def test_list_is_single_chunk() -> None:
    chunks = chunk_markdown(LIST_MD)
    list_chunks = [c for c in chunks if "Item one" in c["text"]]
    assert len(list_chunks) == 1
    assert "Item three" in list_chunks[0]["text"]


def test_empty_paragraphs_skipped() -> None:
    chunks = chunk_markdown(SIMPLE_MD)
    assert all(c["text"].strip() for c in chunks)


COLON_LIST_MD = """# Policy

## Benefits

This policy pays benefits in the event that you:-

✓ Need to cancel your trip; or
✓ Suffer illness or injury; or
✓ Are delayed en route.

Next unrelated paragraph.
"""

SHORT_CHUNK_MD = """# Policy

## Benefits

Short line.

This is a longer paragraph with more content that stands on its own.
"""


def test_colon_intro_merged_with_following_list() -> None:
    chunks = chunk_markdown(COLON_LIST_MD)
    merged = next(c for c in chunks if "cancel your trip" in c["text"])
    assert "pays benefits" in merged["text"]


def test_short_chunk_merged_with_next() -> None:
    chunks = chunk_markdown(SHORT_CHUNK_MD)
    assert not any(len(c["text"]) < 50 and "Short line" in c["text"] for c in chunks)


def test_chunk_numbers_sequential_after_merge() -> None:
    chunks = chunk_markdown(COLON_LIST_MD)
    assert [c["chunk_number"] for c in chunks] == list(range(len(chunks)))


NUMBERED_SECTIONS_MD = """## 2 Travel Benefits

Some intro text.

## 2.1 Travel Exclusions

Exclusions content.

## 2.1.1 Sub Exclusions

Sub exclusion detail.

## 2.2 Another Section

Another section content.
"""


def test_numbered_heading_depth_1() -> None:
    chunks = chunk_markdown(NUMBERED_SECTIONS_MD)
    intro = next(c for c in chunks if "intro text" in c["text"])
    assert intro["section_breadcrumbs"] == ["2 Travel Benefits"]


def test_numbered_heading_depth_2() -> None:
    chunks = chunk_markdown(NUMBERED_SECTIONS_MD)
    chunk = next(c for c in chunks if "Exclusions content" in c["text"])
    assert chunk["section_breadcrumbs"] == ["2 Travel Benefits", "2.1 Travel Exclusions"]


def test_numbered_heading_depth_3() -> None:
    chunks = chunk_markdown(NUMBERED_SECTIONS_MD)
    chunk = next(c for c in chunks if "Sub exclusion detail" in c["text"])
    assert chunk["section_breadcrumbs"] == [
        "2 Travel Benefits",
        "2.1 Travel Exclusions",
        "2.1.1 Sub Exclusions",
    ]


def test_numbered_heading_sibling_clears_deeper_breadcrumbs() -> None:
    chunks = chunk_markdown(NUMBERED_SECTIONS_MD)
    chunk = next(c for c in chunks if "Another section content" in c["text"])
    assert chunk["section_breadcrumbs"] == ["2 Travel Benefits", "2.2 Another Section"]


NON_NUMBERED_MD = """## Introduction

Some intro.

## 2.1 Travel Exclusions

Exclusions content.
"""


def test_non_numbered_heading_not_stacked() -> None:
    """Non-numbered headings still produce a flat single-entry breadcrumb."""
    chunks = chunk_markdown(NON_NUMBERED_MD)
    intro = next(c for c in chunks if "Some intro" in c["text"])
    assert intro["section_breadcrumbs"] == ["Introduction"]


def test_numbered_heading_after_non_numbered_starts_fresh_stack() -> None:
    chunks = chunk_markdown(NON_NUMBERED_MD)
    chunk = next(c for c in chunks if "Exclusions content" in c["text"])
    # MarkdownHeaderTextSplitter tracks nesting: h2 under h2 becomes nested
    assert "2.1 Travel Exclusions" in chunk["section_breadcrumbs"]
