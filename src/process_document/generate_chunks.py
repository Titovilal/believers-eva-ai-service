"""
Generate chunks from text using Recursive Character Text Splitting.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..utils.constants import CHUNKING


def generate_chunks(
    text: str,
    chunk_size: int = CHUNKING["chunk_size"],
    chunk_overlap: int = CHUNKING["chunk_overlap"],
) -> List[str]:
    """
    Generate chunks from text using Recursive Character Text Splitting.

    Args:
        text: Input text to chunk
        chunk_size: Maximum number of characters per chunk (default: 512)
        chunk_overlap: Number of characters to overlap between chunks (default: 0)

    Returns:
        List[str]: List of text chunks

    Raises:
        ValueError: If text is empty or chunk_size is invalid
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    # Create text splitter using character-based recursive splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNKING["separators"],
    )

    # Split text into chunks
    chunks = text_splitter.split_text(text)

    return chunks
