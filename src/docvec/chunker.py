import re

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_PAGE_SEP_RE = re.compile(r"^-{3,}.*$", re.MULTILINE)
_PROMOTE_RE = re.compile(r"^## ((\d+(?:\.\d+)+)\s.*)$", re.MULTILINE)

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"), ("#####", "h5"), ("######", "h6")]


def _promote_numbered_headings(md: str) -> str:
    def _replace(m: re.Match) -> str:
        extra = "#" * m.group(2).count(".")
        return f"##{extra} {m.group(1)}"
    return _PROMOTE_RE.sub(_replace, md)


def _breadcrumbs_from_meta(meta: dict) -> list[str]:
    """Extract ordered breadcrumb list from MarkdownHeaderTextSplitter metadata."""
    return [meta[k] for k in ("h1", "h2", "h3", "h4", "h5", "h6") if k in meta]


def chunk_markdown(md: str, chunk_size: int = 3000, chunk_overlap: int = 0) -> list[dict]:
    """Split markdown into chunks.

    Uses MarkdownHeaderTextSplitter to split on section boundaries first,
    then RecursiveCharacterTextSplitter for any sections that exceed chunk_size.

    Each chunk has: page_number, chunk_number, section_breadcrumbs, text.
    Page number increments on page separators (--- or pymupdf4llm style).
    """
    md = _promote_numbered_headings(md)

    # Split into per-page segments to track page numbers
    pages = _PAGE_SEP_RE.split(md)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strip_whitespace=True,
    )

    chunks: list[dict] = []
    for page_idx, page_text in enumerate(pages):
        page_number = page_idx + 1
        header_docs = header_splitter.split_text(page_text)
        for doc in header_docs:
            text = doc.page_content.strip()
            if not text:
                continue
            crumbs = _breadcrumbs_from_meta(doc.metadata)
            # If this section is too large, split further
            if len(text) > chunk_size:
                sub_docs = char_splitter.create_documents([text])
                for sub in sub_docs:
                    sub_text = sub.page_content.strip()
                    if sub_text:
                        chunks.append({
                            "page_number": page_number,
                            "chunk_number": len(chunks),
                            "section_breadcrumbs": crumbs,
                            "text": sub_text,
                        })
            else:
                chunks.append({
                    "page_number": page_number,
                    "chunk_number": len(chunks),
                    "section_breadcrumbs": crumbs,
                    "text": text,
                })

    return chunks
