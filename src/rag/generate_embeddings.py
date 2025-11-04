"""
Generate embeddings for text chunks using OpenAI.
"""
import os
from typing import List
from openai import OpenAI


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
