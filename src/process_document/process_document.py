"""
Document Processing Module
Handles base64 encoded documents from requests, supports PDF and text files.
"""

import base64
import tempfile
import time
from pathlib import Path

from .parse_pdf import parse_pdf
from .generate_chunks import generate_chunks
from .generate_embeddings import generate_embeddings
from .detect_number_in_text import detect_number_in_text
from .extract_verifiable_data import extract_verifiable_data
from ..utils.constants import (
    PARSE_PDF_DEFAULT_ENABLE_IMAGE_ANNOTATION,
    PARSE_PDF_DEFAULT_FORCE_OCR,
    CHUNK_DEFAULT_SIZE,
    CHUNK_DEFAULT_OVERLAP,
    EMBEDDING_DEFAULT_MODEL,
    LANGUAGE_DEFAULT,
    VERIFIABLE_DEFAULT_EXTRACT,
    VERIFIABLE_DEFAULT_MODEL,
)


async def process_document(
    base64_data: str,
    enable_image_annotation: bool = PARSE_PDF_DEFAULT_ENABLE_IMAGE_ANNOTATION,
    force_ocr: bool = PARSE_PDF_DEFAULT_FORCE_OCR,
    chunk_size: int = CHUNK_DEFAULT_SIZE,
    chunk_overlap: int = CHUNK_DEFAULT_OVERLAP,
    model: str = EMBEDDING_DEFAULT_MODEL,
    lang: str = LANGUAGE_DEFAULT,
    extract_verifiable: bool = VERIFIABLE_DEFAULT_EXTRACT,
    verifiable_model: str = VERIFIABLE_DEFAULT_MODEL,
) -> dict:
    """
    Process a base64 encoded document from a request.

    Args:
        base64_data: Base64 encoded document string (typical from request body)
        enable_image_annotation: If True, annotate images in PDFs with AI descriptions
        force_ocr: If True, force OCR even for PDFs with native text (default: False)
        chunk_size: Maximum number of characters per chunk (default: 512)
        chunk_overlap: Number of characters to overlap between chunks (default: 0)
        model: OpenAI embedding model (default: "text-embedding-3-small")
        lang: Language code for number detection (default: "es")
        extract_verifiable: If True, extract verifiable data from chunks (default: True)
        verifiable_model: OpenAI chat model for verifiable data extraction (default: "gpt-5-mini")

    Returns:
        dict: Document content and metadata including:
            - text: Extracted text from the document
            - chunks: List of text chunks
            - embeddings: List of embedding vectors for each chunk
            - chunks_with_numbers: List of booleans indicating which chunks contain numbers
            - chunk_count: Total number of chunks
            - verifiable_data: Extracted verifiable data (if extract_verifiable=True)
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
        result = _parse_pdf(decoded_data, enable_image_annotation, force_ocr)
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
    chunks = generate_chunks(
        text_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Detect which chunks contain numbers
    chunks_with_numbers = [detect_number_in_text(chunk, lang) for chunk in chunks]

    # Generate embeddings for chunks
    embeddings, embeddings_usage = await generate_embeddings(chunks, model=model)

    # Add processing results to the result dictionary
    result["chunks"] = chunks
    result["embeddings"] = embeddings
    result["chunks_with_numbers"] = chunks_with_numbers
    result["chunk_count"] = len(chunks)

    # Initialize verifiable data costs
    verifiable_usage = {}

    # Extract verifiable data if enabled
    if extract_verifiable:
        verifiable_result = await extract_verifiable_data(
            chunks, chunks_with_numbers, model=verifiable_model
        )
        verifiable_usage = verifiable_result.get("usage", {})
        result["verifiable_data"] = _filter_verifiable_statements_with_numbers(
            verifiable_result, lang
        )

    # Calculate processing time
    processing_time = time.time() - start_time

    # Consolidate all costs and metrics
    total_cost = (
        pdf_usage.get("vision_cost", 0.0)
        + embeddings_usage.get("cost", 0.0)
        + verifiable_usage.get("cost", 0.0)
    )

    result["processing_metrics"] = {
        "processing_time_seconds": round(processing_time, 2),
        "total_chunks": len(chunks),
        "costs": {
            "image_annotation": {
                "images_processed": pdf_usage.get("images_annotated", 0),
                "tokens_estimated": pdf_usage.get("vision_tokens_estimated", 0),
                "cost": pdf_usage.get("vision_cost", 0.0),
            },
            "ocr": {
                "used": pdf_usage.get("ocr_used", False),
                "cost": 0.0,  # OCR is included in Docling, no additional cost
            },
            "embeddings": {
                "tokens": embeddings_usage.get("tokens", 0),
                "model": embeddings_usage.get("model", model),
                "cost": embeddings_usage.get("cost", 0.0),
            },
            "verifiable_data": {
                "input_tokens": verifiable_usage.get("input_tokens", 0),
                "output_tokens": verifiable_usage.get("output_tokens", 0),
                "model": verifiable_usage.get("model", verifiable_model),
                "cost": verifiable_usage.get("cost", 0.0),
            },
            "total_cost": round(total_cost, 6),
        },
    }

    # TODO: Add database upload step
    # Upload processed document data to database:
    # - Document metadata (file_type, page_count, etc.)
    # - Text chunks with embeddings
    # - Verifiable data extracted
    # - Link chunks with their corresponding embeddings and numeric flags

    return result


def _filter_verifiable_statements_with_numbers(
    verifiable_result: dict, lang: str
) -> dict:
    """
    Filter verifiable statements to only include those containing numbers.

    Args:
        verifiable_result: Result from extract_verifiable_data
        lang: Language code for number detection

    Returns:
        dict: Filtered result with summary and verifiable_data
    """
    filtered_verifiable_data = []
    for fact_group in verifiable_result["verifiable_data"]:
        if "statements" in fact_group and fact_group["statements"]:
            # Filter statements that contain numbers
            statements_with_numbers = [
                stmt
                for stmt in fact_group["statements"]
                if detect_number_in_text(stmt, lang)
            ]

            if statements_with_numbers:
                filtered_fact_group = fact_group.copy()
                filtered_fact_group["statements"] = statements_with_numbers
                filtered_verifiable_data.append(filtered_fact_group)

    return {
        "verifiable_data": filtered_verifiable_data,
    }


def _parse_pdf(
    pdf_data: bytes,
    enable_image_annotation: bool = PARSE_PDF_DEFAULT_ENABLE_IMAGE_ANNOTATION,
    force_ocr: bool = PARSE_PDF_DEFAULT_FORCE_OCR,
) -> dict:
    """
    Process PDF file data.

    Args:
        pdf_data: Raw PDF file bytes
        enable_image_annotation: If True, annotate images with AI descriptions
        force_ocr: If True, force OCR even for PDFs with native text

    Returns:
        dict: Parsed PDF content and metadata
    """
    # Create a temporary file to store the PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_data)
        temp_path = temp_file.name

    try:
        # Parse the PDF using the parse_pdf function
        result = parse_pdf(temp_path, enable_image_annotation, force_ocr)
        result["file_type"] = "pdf"
        return result
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink(missing_ok=True)
