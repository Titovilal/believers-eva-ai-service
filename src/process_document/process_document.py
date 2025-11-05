"""
Document Processing Module
Handles base64 encoded documents from requests, supports PDF and text files.
"""

import base64
import tempfile
from pathlib import Path

from .parse_pdf import parse_pdf
from .generate_chunks import generate_chunks
from .generate_embeddings import generate_embeddings
from .detect_number_in_text import detect_number_in_text
from .extract_verifiable_data import extract_verifiable_data
from ..utils.constants import (
    DEFAULT_ENABLE_IMAGE_ANNOTATION,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_EXTRACT_VERIFIABLE,
    DEFAULT_VERIFIABLE_MODEL,
)


def process_document(
    base64_data: str,
    enable_image_annotation: bool = DEFAULT_ENABLE_IMAGE_ANNOTATION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    model: str = DEFAULT_EMBEDDING_MODEL,
    lang: str = DEFAULT_LANGUAGE,
    extract_verifiable: bool = DEFAULT_EXTRACT_VERIFIABLE,
    verifiable_model: str = DEFAULT_VERIFIABLE_MODEL,
) -> dict:
    """
    Process a base64 encoded document from a request.

    Args:
        base64_data: Base64 encoded document string (typical from request body)
        enable_image_annotation: If True, annotate images in PDFs with AI descriptions
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
            - verifiable_facts: Extracted verifiable data (if extract_verifiable=True)
            - file_type: 'pdf' or 'text'
            - (For PDFs) metadata, page_count, file_name

    Raises:
        ValueError: If the file is neither PDF nor text, or if base64 is invalid
    """
    try:
        # Decode base64 data
        decoded_data = base64.b64decode(base64_data)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {str(e)}")

    # Check if it's a PDF by looking at the file signature
    if decoded_data.startswith(b"%PDF"):
        result = _process_pdf(decoded_data, enable_image_annotation)
        text_content = result["text"]
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

    # Generate embeddings for chunks
    embeddings = generate_embeddings(chunks, model=model)

    # Detect which chunks contain numbers
    chunks_with_numbers = [detect_number_in_text(chunk, lang) for chunk in chunks]

    # Add processing results to the result dictionary
    result["chunks"] = chunks
    result["embeddings"] = embeddings
    result["chunks_with_numbers"] = chunks_with_numbers
    result["chunk_count"] = len(chunks)

    # Extract verifiable data if enabled
    if extract_verifiable:
        verifiable_result = extract_verifiable_data(
            chunks, chunks_with_numbers, model=verifiable_model
        )
        result["verifiable_facts"] = _filter_verifiable_statements_with_numbers(
            verifiable_result, lang
        )

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
        dict: Filtered result with summary and verifiable_facts
    """
    filtered_verifiable_facts = []
    for fact_group in verifiable_result["verifiable_facts"]:
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
                filtered_verifiable_facts.append(filtered_fact_group)

    return {
        "summary": {
            "total_chunks_analyzed": verifiable_result["summary"][
                "total_chunks_analyzed"
            ],
            "total_statements_extracted": sum(
                len(fg["statements"]) for fg in filtered_verifiable_facts
            ),
        },
        "verifiable_facts": filtered_verifiable_facts,
    }


def _process_pdf(
    pdf_data: bytes, enable_image_annotation: bool = DEFAULT_ENABLE_IMAGE_ANNOTATION
) -> dict:
    """
    Process PDF file data.

    Args:
        pdf_data: Raw PDF file bytes
        enable_image_annotation: If True, annotate images with AI descriptions

    Returns:
        dict: Parsed PDF content and metadata
    """
    # Create a temporary file to store the PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_data)
        temp_path = temp_file.name

    try:
        # Parse the PDF using the parse_pdf function
        result = parse_pdf(temp_path, enable_image_annotation)
        result["file_type"] = "pdf"
        return result
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink(missing_ok=True)
