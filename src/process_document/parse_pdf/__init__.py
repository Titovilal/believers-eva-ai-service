"""
PDF Parser Module
Main orchestrator for parsing and extracting text from PDF files.
"""

from pathlib import Path
from .parse_pdf_with_docling import parse_pdf_with_docling
from .parse_pdf_with_raw_openai import parse_pdf_with_raw_openai
from ...utils.constants import PARSE_PDF_DEFAULT_IMAGE_DETAIL


def parse_pdf(
    pdf_path: str | Path,
    enable_image_annotation: bool = False,
    force_ocr: bool = False,
    image_detail: str = PARSE_PDF_DEFAULT_IMAGE_DETAIL,
) -> dict:
    """Parse a PDF file and extract all text content in a markdown format.

    Args:
        pdf_path: Path to the PDF file
        enable_image_annotation: Whether to enable image annotation (only for Docling)
        force_ocr: Whether to force OCR using OpenAI Vision API
        image_detail: Detail level for images - "low" (70 tokens) or "high" (~630 tokens)

    Returns:
        dict with keys: text, page_count, usage
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"File must be a PDF: {pdf_path}")

    if force_ocr:
        return parse_pdf_with_raw_openai(pdf_path, image_detail)

    return parse_pdf_with_docling(pdf_path, enable_image_annotation, image_detail)
