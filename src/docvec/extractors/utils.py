import fitz  # PyMuPDF

from docvec.extractors.models import ScannedPDFError

# Minimum total characters across sampled pages to consider a PDF digital.
_SCAN_TEXT_THRESHOLD = 100
# Number of pages to sample from the start of the document.
_SCAN_SAMPLE_PAGES = 3


def detect_scanned(pdf_path: str) -> None:
    """Raise ScannedPDFError if the PDF appears to be image-only (scanned).

    Samples the first few pages with PyMuPDF and checks total extractable text.
    A digital PDF will have plenty of text; a scanned one will have none.

    Args:
        pdf_path: Path to the PDF file.

    Raises:
        ScannedPDFError: If total text across sampled pages is below threshold.
    """
    with fitz.open(pdf_path) as doc:
        sample = min(_SCAN_SAMPLE_PAGES, len(doc))
        total_text = "".join(doc.load_page(i).get_text() for i in range(sample))

    if len(total_text) < _SCAN_TEXT_THRESHOLD:
        raise ScannedPDFError(
            f"PDF appears to be scanned: only {len(total_text)} characters extracted "
            f"from the first {sample} page(s). OCR is required to process this document."
        )
