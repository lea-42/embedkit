import pytest
from docvec.extractor import _promote_numbered_headings


def test_no_numbered_prefix_unchanged() -> None:
    md = "## Introduction\n\nSome text."
    assert _promote_numbered_headings(md) == md


def test_top_level_number_unchanged() -> None:
    """'2 Travel Benefits' has 0 dots — stays at ##."""
    md = "## 2 Travel Benefits\n\nText."
    assert _promote_numbered_headings(md) == md


def test_one_dot_adds_one_hash() -> None:
    md = "## 2.1 Travel Exclusions\n\nText."
    assert _promote_numbered_headings(md) == "### 2.1 Travel Exclusions\n\nText."


def test_two_dots_adds_two_hashes() -> None:
    md = "## 2.1.3 Sub Section\n\nText."
    assert _promote_numbered_headings(md) == "#### 2.1.3 Sub Section\n\nText."


def test_non_heading_line_unchanged() -> None:
    md = "2.1 this is just paragraph text"
    assert _promote_numbered_headings(md) == md


def test_mixed_document() -> None:
    md = (
        "## 2 Travel Benefits\n\n"
        "Intro.\n\n"
        "## 2.1 Travel Exclusions\n\n"
        "Exclusions.\n\n"
        "## 2.1.1 Sub Exclusions\n\n"
        "Detail.\n\n"
        "## Introduction\n\n"
        "Not numbered."
    )
    expected = (
        "## 2 Travel Benefits\n\n"
        "Intro.\n\n"
        "### 2.1 Travel Exclusions\n\n"
        "Exclusions.\n\n"
        "#### 2.1.1 Sub Exclusions\n\n"
        "Detail.\n\n"
        "## Introduction\n\n"
        "Not numbered."
    )
    assert _promote_numbered_headings(md) == expected
