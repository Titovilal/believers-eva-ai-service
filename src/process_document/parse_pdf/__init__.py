"""PDF Parser Module
Main orchestrator for parsing and extracting text from PDF files.
"""

import tempfile
from pathlib import Path
from .parse_pdf_with_docling import parse_pdf_with_docling
from .parse_pdf_with_raw_openai import parse_pdf_with_raw_openai
from ...utils.constants import PARSE_PDF


def parse_pdf(
    pdf_input: str | Path | bytes,
    enable_image_annotation: bool = False,
    force_ocr: bool = False,
    image_detail: str = PARSE_PDF["docling"]["image_detail"],
) -> dict:
    """Parse a PDF file and extract all text content in a markdown format.

    Args:
        pdf_input: Path to the PDF file (str/Path) or raw PDF bytes data
        enable_image_annotation: Whether to enable image annotation (only for Docling)
        force_ocr: Whether to force OCR using OpenAI Vision API
        image_detail: Detail level for images - "low" (70 tokens) or "high" (~630 tokens)

    Returns:
        dict with keys: text, page_count, usage, file_type
    """
    # Handle bytes input by creating a temporary file
    if isinstance(pdf_input, bytes):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_input)
            temp_path = Path(temp_file.name)

        try:
            result = _parse_pdf_file(
                temp_path, enable_image_annotation, force_ocr, image_detail
            )
            result["file_type"] = "pdf"
            return result
        finally:
            temp_path.unlink(missing_ok=True)

    # Handle path input
    pdf_path = Path(pdf_input)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"File must be a PDF: {pdf_path}")

    return _parse_pdf_file(pdf_path, enable_image_annotation, force_ocr, image_detail)


def _parse_pdf_file(
    pdf_path: Path,
    enable_image_annotation: bool,
    force_ocr: bool,
    image_detail: str,
) -> dict:
    """Internal function to parse a PDF file from a path."""
    if force_ocr:
        return parse_pdf_with_raw_openai(pdf_path, image_detail)

    return parse_pdf_with_docling(pdf_path, enable_image_annotation, image_detail)
