from pydantic import BaseModel


class ScannedPDFError(Exception):
    """Raised when a PDF appears to be scanned (image-only, no extractable text)."""
    pass


# ---------------------------------------------------------------------------
# Table extraction models (used by table_extractor and pymupdf_extractor)
# ---------------------------------------------------------------------------

class TableCell(BaseModel):
    row: int  # 0-based, row 0 is header
    col: int  # 0-based
    text: str


class Table(BaseModel):
    caption: str | None = None
    cells: list[TableCell]


class PageTables(BaseModel):
    tables: list[Table]


# ---------------------------------------------------------------------------
# OpenAI-facing models (lean schemas sent to instructor / OpenAI)
# ---------------------------------------------------------------------------

class ImageRefBase(BaseModel):
    """Image reference as extracted by the LLM — no binary data."""
    page_start: int  # 1-based page number where the image appears
    caption: str | None = None
    description: str | None = None


class TableBase(BaseModel):
    caption: str | None = None
    cells: list[TableCell] = []


class SectionBase(BaseModel):
    heading: str
    level: int  # 1 = top-level, 2 = sub, etc. based on document structure
    page_start: int  # 1-based page number this section starts on
    body: str = ""
    tables: list[TableBase] = []
    images: list[ImageRefBase] = []


class BatchExtractionBase(BaseModel):
    """Output of one batch call — sections across all pages in the batch."""
    sections: list[SectionBase] = []
    open_sections: list[str] = []  # full heading path still open at batch end


class DocumentExtractionBase(BaseModel):
    """Merged result across all batches."""
    sections: list[SectionBase] = []


