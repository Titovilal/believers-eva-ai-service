"""
Document Processing Module
Handles base64 encoded documents from requests, supports PDF and text files.
"""

import base64
import time

from .parse_pdf import parse_pdf
from .detect_number_in_text import detect_number_in_text
from .generate_chunks import generate_chunks
from .generate_embeddings import generate_embeddings
from .extract_verifiable_data import extract_verifiable_data
from ..utils.constants import DETECT_NUMBERS


async def process_document(
    base64_data: str,
    enable_image_annotation: bool = False,
    force_ocr: bool = False,
    lang: str = DETECT_NUMBERS["language"],
    extract_verifiable: bool = True,
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

    # Generate chunks from text
    chunks = generate_chunks(text_content)

    # Detect numbers in chunks
    chunks_with_numbers = [detect_number_in_text(chunk, lang) for chunk in chunks]

    # Generate embeddings
    embeddings, embeddings_usage = await generate_embeddings(chunks)

    # Extract verifiable data if requested
    verifiable_usage = {}
    if extract_verifiable:
        verifiable_result = await extract_verifiable_data(chunks, chunks_with_numbers)
        result["verifiable_data"] = verifiable_result
        verifiable_usage = verifiable_result["usage"]

    # Add chunks and embeddings to result
    result["chunks"] = chunks
    result["chunks_with_numbers"] = chunks_with_numbers
    result["embeddings"] = embeddings

    # Calculate processing time
    processing_time = time.time() - start_time

    # Consolidate all costs and metrics
    total_cost = (
        pdf_usage.get("cost", 0.0)
        + embeddings_usage.get("cost", 0.0)
        + verifiable_usage.get("cost", 0.0)
    )

    result["processing_metrics"] = {
        "processing_time_seconds": round(processing_time, 2),
        "total_chunks": len(chunks),
        "costs": {
            "parse_pdf": {
                "input_tokens": pdf_usage.get("input_tokens", 0),
                "output_tokens": pdf_usage.get("output_tokens", 0),
                "model": pdf_usage.get("model", ""),
                "cost": pdf_usage.get("cost", 0.0),
            },
            "embeddings": {
                "input_tokens": embeddings_usage.get("input_tokens", 0),
                "model": embeddings_usage.get("model", ""),
                "cost": embeddings_usage.get("cost", 0.0),
            },
            "verifiable_data": {
                "input_tokens": verifiable_usage.get("input_tokens", 0),
                "output_tokens": verifiable_usage.get("output_tokens", 0),
                "model": verifiable_usage.get("model", ""),
                "cost": verifiable_usage.get("cost", 0.0),
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
