"""
Extract verifiable data from text chunks using AI analysis.
"""

import os
import json
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI

from ..utils.constants import (
    VERIFIABLE_DEFAULT_MODEL,
    VERIFIABLE_DEFAULT_BATCH_SIZE,
    VERIFIABLE_MODEL_INPUT_PRICE,
    VERIFIABLE_MODEL_OUTPUT_PRICE,
)


async def _process_batch_async(
    client: AsyncOpenAI,
    batch: List[tuple],
    model: str,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """Process a batch of chunks asynchronously."""
    try:
        # Prepare the content with all chunks in the batch
        batch_content = "Analyze the following text chunks and extract verifiable data from each one:\n\n"
        for i, (chunk_index, chunk_text) in enumerate(batch):
            batch_content += f"--- CHUNK {chunk_index} ---\n{chunk_text}\n\n"

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batch_content},
            ],
            response_format={"type": "json_object"},
        )

        extracted_json = json.loads(response.choices[0].message.content)
        usage = response.usage

        # Process the response for each chunk
        results = []
        chunks_data = extracted_json.get("chunks", [])

        for i, (chunk_index, _) in enumerate(batch):
            chunk_data = chunks_data[i] if i < len(chunks_data) else {}
            results.append(
                {
                    "chunk_index": chunk_index,
                    "statements": chunk_data.get("statements", []),
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens // len(batch),
                        "completion_tokens": usage.completion_tokens // len(batch),
                        "total_tokens": usage.total_tokens // len(batch),
                    },
                }
            )

        return results

    except Exception as e:
        # Log error but continue processing other chunks
        return [
            {
                "chunk_index": chunk_index,
                "statements": [],
                "error": str(e),
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            for chunk_index, _ in batch
        ]


async def _extract_verifiable_data_async(
    chunks_to_analyze: List[tuple],
    model: str,
    api_key: str,
    batch_size: int = VERIFIABLE_DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """Process chunks in batches to reduce API calls."""
    client = AsyncOpenAI(api_key=api_key)

    system_prompt = """You are an expert assistant in extracting verifiable and factual data from texts.
Your task is to analyze text fragments and extract short phrases that contain verifiable information such as:
- Dates, numbers, quantities, percentages
- Metrics, statistics, measurements
- Claims with concrete data

Extract short phrases (maximum 15-20 words) that contain this verifiable data.
Each phrase should be concise but maintain enough context to be understood.

You will receive multiple text chunks. For each chunk, extract the verifiable data.

Respond ONLY with a valid JSON object with this structure:
{
    "chunks": [
        {"statements": ["phrase with verifiable data 1", "phrase with verifiable data 2", ...]},
        {"statements": ["phrase with verifiable data 3", "phrase with verifiable data 4", ...]},
        ...
    ]
}

Return one object per chunk in the same order. If you don't find verifiable data in a chunk, return an empty array for that chunk."""

    # Group chunks into batches
    batches = [
        chunks_to_analyze[i : i + batch_size]
        for i in range(0, len(chunks_to_analyze), batch_size)
    ]

    # Create tasks for all batches
    tasks = [
        _process_batch_async(client, batch, model, system_prompt) for batch in batches
    ]

    # Process all batches in parallel
    batch_results = await asyncio.gather(*tasks)

    # Flatten results from all batches
    results = []
    for batch_result in batch_results:
        results.extend(batch_result)

    return results


async def extract_verifiable_data(
    chunks: List[str],
    chunks_with_numbers: List[bool],
    model: str = VERIFIABLE_DEFAULT_MODEL,
    batch_size: int = VERIFIABLE_DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Analyze chunks containing numbers and extract verifiable data using AI.

    Args:
        chunks: List of text chunks
        chunks_with_numbers: Boolean list indicating which chunks contain numbers
        model: OpenAI chat model to use for analysis
        batch_size: Number of chunks to process in each batch (default: 5)

    Returns:
        dict: JSON structure containing extracted verifiable data with format:
            {
                "verifiable_data": [
                    {
                        "chunk_index": int,
                        "statements": ["phrase with verifiable data 1", "phrase with verifiable data 2", ...]
                    },
                    ...
                ],
                "summary": {
                    "total_chunks_analyzed": int,
                    "total_statements_extracted": int
                },
                "usage": {
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "total_tokens": int,
                    "cost": float
                }
            }

    Raises:
        ValueError: If chunks and chunks_with_numbers lengths don't match or OPENAI_API_KEY is missing
        Exception: If there's an error during AI analysis
    """
    if len(chunks) != len(chunks_with_numbers):
        raise ValueError("Chunks and chunks_with_numbers must have the same length")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Filter chunks that contain numbers
    chunks_to_analyze = [
        (i, chunk)
        for i, (chunk, has_number) in enumerate(zip(chunks, chunks_with_numbers))
        if has_number
    ]

    if not chunks_to_analyze:
        return {
            "verifiable_data": [],
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "model": model,
                "cost": 0.0,
            },
        }

    try:
        # Run async processing
        verifiable_data = await _extract_verifiable_data_async(
            chunks_to_analyze, model, api_key, batch_size
        )

        # Count total tokens
        total_prompt_tokens = sum(
            item.get("usage", {}).get("prompt_tokens", 0) for item in verifiable_data
        )
        total_completion_tokens = sum(
            item.get("usage", {}).get("completion_tokens", 0)
            for item in verifiable_data
        )

        # Calculate cost using model pricing from constants
        cost = (
            total_prompt_tokens * VERIFIABLE_MODEL_INPUT_PRICE
            + total_completion_tokens * VERIFIABLE_MODEL_OUTPUT_PRICE
        ) / 1_000_000

        return {
            "verifiable_data": verifiable_data,
            "usage": {
                "input_tokens": total_prompt_tokens,
                "output_tokens": total_completion_tokens,
                "model": model,
                "cost": cost,
            },
        }

    except Exception as e:
        raise Exception(f"Error extracting verifiable data: {str(e)}")
