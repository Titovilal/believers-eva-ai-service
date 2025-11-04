"""
Generate chunks from text using Recursive Character Text Splitting.
"""
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def generate_chunks(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
    model: str = "text-embedding-3-small"
) -> List[str]:
    """
    Generate chunks from text using Recursive Character Text Splitting.

    Args:
        text: Input text to chunk
        chunk_size: Maximum number of characters per chunk (default: 512)
        chunk_overlap: Number of characters to overlap between chunks (default: 0)
        model: OpenAI model for embeddings (not used in chunking)

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
        separators=["\n\n", "\n", " ", ""]
    )

    # Split text into chunks
    chunks = text_splitter.split_text(text)

    return chunks
