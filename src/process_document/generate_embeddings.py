"""
Generate embeddings for text chunks using OpenAI.
"""

import os
from typing import List
from openai import AsyncOpenAI
from ..utils.constants import EMBEDDINGS


async def generate_embeddings(
    chunks: List[str], model: str = EMBEDDINGS["model_id"]
) -> tuple[List[List[float]], dict]:
    """
    Generate embeddings for a list of text chunks using OpenAI.

    Args:
        chunks: List of text chunks to embed
        model: OpenAI embedding model (must be "text-embedding-3-small")

    Returns:
        tuple: (embeddings, usage_info) where:
            - embeddings: List of embedding vectors
            - usage_info: Dict with 'input_tokens' and 'cost' information

    Raises:
        ValueError: If chunks is empty, model is not text-embedding-3-small, or OPENAI_API_KEY is missing
        Exception: If there's an error generating embeddings
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")

    if model != "text-embedding-3-small":
        raise ValueError(
            f"Only 'text-embedding-3-small' model is supported, got '{model}'"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    try:
        client = AsyncOpenAI(api_key=api_key)

        # Generate embeddings for all chunks
        response = await client.embeddings.create(input=chunks, model=model)

        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]

        # Calculate cost
        tokens_used = response.usage.total_tokens

        usage_info = {
            "input_tokens": tokens_used,
            "cost": tokens_used * EMBEDDINGS["model_price"] / EMBEDDINGS["model_pricing_unit"],
            "model": model,
        }

        return embeddings, usage_info

    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")
