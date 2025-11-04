"""
RAG Database Module
Functions for chunking text and generating embeddings.
"""
import os
import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from text_to_num import text2num



def generate_chunks_from_text(
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


def generate_embeddings(
    chunks: List[str],
    model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using OpenAI.

    Args:
        chunks: List of text chunks to embed
        model: OpenAI embedding model (default: "text-embedding-3-small")

    Returns:
        List[List[float]]: List of embedding vectors

    Raises:
        ValueError: If chunks is empty or OPENAI_API_KEY is missing
        Exception: If there's an error generating embeddings
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    try:
        client = OpenAI(api_key=api_key)

        # Generate embeddings for all chunks
        response = client.embeddings.create(
            input=chunks,
            model=model
        )

        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]

        return embeddings

    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")


def contains_number(text: str, lang: str) -> bool:
    """
    Check if text contains any number in digit or text format.

    Args:
        text: Input text to check
        lang: Language code for text-to-number conversion 

    Returns:
        bool: True if text contains numbers (digit or text), False otherwise
    """
    if not text:
        return False

    # Check for digit numbers using regex
    if re.search(r'\d+', text):
        return True

    # Check for text numbers using text2num
    words = text.lower().split()
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word)
        try:
            text2num(clean_word, lang)
            return True
        except ValueError:
            pass

    return False

