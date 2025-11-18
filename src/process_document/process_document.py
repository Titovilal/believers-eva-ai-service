"""
Document Processing Module
Handles base64 encoded documents from requests, supports PDF and text files.
"""

import base64
import time

from .parse_pdf import parse_pdf
from .detect_number_in_text import detect_number_in_text
from ..utils.constants_old import (
    PARSE_PDF_DEFAULT_ENABLE_IMAGE_ANNOTATION,
    PARSE_PDF_DEFAULT_FORCE_OCR,
    LANGUAGE_DEFAULT,
    VERIFIABLE_DEFAULT_EXTRACT,
)


async def process_document(
    base64_data: str,
    enable_image_annotation: bool = PARSE_PDF_DEFAULT_ENABLE_IMAGE_ANNOTATION,
    force_ocr: bool = PARSE_PDF_DEFAULT_FORCE_OCR,
    lang: str = LANGUAGE_DEFAULT,
    extract_verifiable: bool = VERIFIABLE_DEFAULT_EXTRACT,
) -> dict:
    """
    Process a base64 encoded document from a request.

    Args:
        base64_data: Base64 encoded document string (typical from request body)
        enable_image_annotation: If True, annotate images in PDFs with AI descriptions
        force_ocr: If True, force OCR even for PDFs with native text (default: False)
        extract_verifiable: If True, extract verifiable data from chunks (default: True)
        lang: Language code for number detection (default: 'en')

    Returns:
        dict: Document content and metadata including:
            - text: Extracted text from the document
            - file_type: 'pdf' or 'text'
            - (For PDFs) metadata, page_count, file_name
            - processing_metrics: Dict with costs and processing time

    Raises:
        ValueError: If the file is neither PDF nor text, or if base64 is invalid
    """
    # Track processing time
    start_time = time.time()

    try:
        # Decode base64 data
        decoded_data = base64.b64decode(base64_data)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {str(e)}")

    # Initialize cost tracking
    pdf_usage = {}

    # Check if it's a PDF by looking at the file signature
    if decoded_data.startswith(b"%PDF"):
        result = parse_pdf(decoded_data, enable_image_annotation, force_ocr)
        text_content = result["text"]
        pdf_usage = result.get("usage", {})
    else:
        # Try to decode as text
        try:
            text_content = decoded_data.decode("utf-8")
            result = {"text": text_content, "file_type": "text"}
        except UnicodeDecodeError:
            # If not PDF and not valid UTF-8 text, raise error
            raise ValueError(
                "Unsupported file type. Only PDF and text files are supported."
            )

    # Calculate processing time
    processing_time = time.time() - start_time

    # Consolidate all costs and metrics
    total_cost = (
        pdf_usage.get("cost", 0.0)
        + 0.0  # embeddings_usage.get("cost", 0.0)
        + 0.0  # verifiable_usage.get("cost", 0.0)
    )

    result["processing_metrics"] = {
        "processing_time_seconds": round(processing_time, 2),
        "total_chunks": 0,  # len(chunks)
        "costs": {
            "parse_pdf": {
                "input_tokens": pdf_usage.get("input_tokens", 0),
                "output_tokens": pdf_usage.get("output_tokens", 0),
                "model": pdf_usage.get("model", ""),
                "cost": pdf_usage.get("cost", 0.0),
            },
            "embeddings": {
                "input_tokens": 0,  # embeddings_usage.get("input_tokens", 0),
                "model": "",  # EMBEDDING_DEFAULT_MODEL,
                "cost": 0.0,  # embeddings_usage.get("cost", 0.0),
            },
            "verifiable_data": {
                "input_tokens": 0,  # verifiable_usage.get("input_tokens", 0),
                "output_tokens": 0,  # verifiable_usage.get("output_tokens", 0),
                "model": "",  # VERIFIABLE_DEFAULT_MODEL,
                "cost": 0.0,  # verifiable_usage.get("cost", 0.0),
            },
            "total_cost": round(total_cost, 3),
        },
    }

    # TODO: Add database upload step
    # Upload processed document data to database:
    # - Document metadata (file_type, page_count, etc.)
    # - Text chunks with embeddings
    # - Verifiable data extracted
    # - Link chunks with their corresponding embeddings and numeric flags

    return result
