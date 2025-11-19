"""
Verifiable Data Extractor using OpenAI Chat API
Analyzes text chunks and extracts verifiable statements containing factual data.
"""

import asyncio
import os
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from ..utils.constants import VERIFIABLE_DATA, DETECT_NUMBERS
from .detect_number_in_text import detect_number_in_text

VERIFIABLE_CONFIG = VERIFIABLE_DATA
OPENAI_SYSTEM_PROMPT = """You are an expert assistant in extracting verifiable and factual data from texts.
Your task is to analyze text fragments and extract short phrases that contain verifiable information such as:
- Dates, numbers, quantities, percentages
- Metrics, statistics, measurements
- Claims with concrete data

Extract short phrases (maximum 15-20 words) that contain this verifiable data.
Each phrase should be concise but maintain enough context to be understood.

You will receive multiple text chunks. For each chunk, extract the verifiable data.
Return one object per chunk in the same order. If you don't find verifiable data in a chunk, return an empty array for that chunk."""


class ChunkStatements(BaseModel):
    statements: list[str]


class VerifiableDataResponse(BaseModel):
    chunks: list[ChunkStatements]


def _build_batch_content(batch: list[tuple]) -> str:
    """Build the batch content for OpenAI Chat API."""
    content = "Analyze the following text chunks and extract verifiable data from each one:\n\n"
    for chunk_index, chunk_text in batch:
        content += f"--- CHUNK {chunk_index} ---\n{chunk_text}\n\n"
    return content


def _parse_batch_response(
    response, batch: list[tuple], lang: str
) -> list[dict[str, Any]]:
    """Parse the response from OpenAI and format results."""
    parsed_output = response.output_parsed
    usage = response.usage
    chunks_data = parsed_output.chunks

    results = []
    for i, (chunk_index, _) in enumerate(batch):
        chunk_data = chunks_data[i] if i < len(chunks_data) else None
        raw_statements = chunk_data.statements if chunk_data else []

        # Filter statements that contain numbers
        filtered_statements = [
            statement
            for statement in raw_statements
            if detect_number_in_text(statement, lang)
        ]

        results.append(
            {
                "chunk_index": chunk_index,
                "statements": filtered_statements,
                "usage": {
                    "prompt_tokens": usage.input_tokens // len(batch),
                    "completion_tokens": usage.output_tokens // len(batch),
                    "total_tokens": (usage.input_tokens + usage.output_tokens)
                    // len(batch),
                },
            }
        )
    return results


def _create_error_results(batch: list[tuple], error: str) -> list[dict[str, Any]]:
    """Create error results for failed batch processing."""
    return [
        {
            "chunk_index": chunk_index,
            "statements": [],
            "error": error,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        for chunk_index, _ in batch
    ]


def _calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost based on token usage."""
    return (
        input_tokens * VERIFIABLE_CONFIG["model_input_price"]
        + output_tokens * VERIFIABLE_CONFIG["model_output_price"]
    ) / VERIFIABLE_CONFIG["model_pricing_unit"]


async def _process_batch_async(
    client: AsyncOpenAI, batch: list[tuple], model: str, reasoning_effort: str, lang: str
) -> list[dict[str, Any]]:
    """Process a batch of chunks asynchronously."""
    try:
        batch_content = _build_batch_content(batch)

        response = await client.responses.parse(
            model=model,
            reasoning={"effort": reasoning_effort},
            input=[
                {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                {"role": "user", "content": batch_content},
            ],
            text_format=VerifiableDataResponse,
        )

        return _parse_batch_response(response, batch, lang)

    except Exception as exc:
        return _create_error_results(batch, str(exc))


async def _extract_verifiable_data_async(
    chunks_to_analyze: list[tuple],
    model: str,
    api_key: str,
    batch_size: int,
    reasoning_effort: str,
    lang: str,
) -> list[dict[str, Any]]:
    """Process chunks in batches to reduce API calls."""
    client = AsyncOpenAI(api_key=api_key)

    batches = [
        chunks_to_analyze[i : i + batch_size]
        for i in range(0, len(chunks_to_analyze), batch_size)
    ]

    tasks = [
        _process_batch_async(client, batch, model, reasoning_effort, lang)
        for batch in batches
    ]
    batch_results = await asyncio.gather(*tasks)

    results = []
    for batch_result in batch_results:
        results.extend(batch_result)

    return results


async def extract_verifiable_data(
    chunks: list[str],
    chunks_with_numbers: list[bool],
    model: str = VERIFIABLE_CONFIG["model_id"],
    batch_size: int = VERIFIABLE_CONFIG["batch_size"],
    reasoning_effort: str = VERIFIABLE_CONFIG["reasoning"],
    lang: str = DETECT_NUMBERS["language"],
) -> dict[str, Any]:
    """Analyze chunks containing numbers and extract verifiable data using AI.

    Args:
        chunks: List of text chunks
        chunks_with_numbers: Boolean list indicating which chunks contain numbers
        model: OpenAI chat model to use for analysis
        batch_size: Number of chunks to process in each batch
        reasoning_effort: Reasoning effort level - "minimal", "low", "medium", "high"
        lang: Language code for number detection (default: 'en')

    Returns:
        Dictionary with verifiable_data and usage information
    """
    if len(chunks) != len(chunks_with_numbers):
        raise ValueError("Chunks and chunks_with_numbers must have the same length")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    chunks_to_analyze = [
        (i, chunk)
        for i, (chunk, has_number) in enumerate(zip(chunks, chunks_with_numbers))
        if has_number
    ]

    if not chunks_to_analyze:
        usage = dict(
            input_tokens=0,
            output_tokens=0,
            model=model,
            cost=0.0,
        )
        return dict(verifiable_data=[], usage=usage)

    try:
        verifiable_data = await _extract_verifiable_data_async(
            chunks_to_analyze, model, api_key, batch_size, reasoning_effort, lang
        )

        total_input_tokens = sum(
            item.get("usage", {}).get("prompt_tokens", 0) for item in verifiable_data
        )
        total_output_tokens = sum(
            item.get("usage", {}).get("completion_tokens", 0)
            for item in verifiable_data
        )

        cost = _calculate_cost(total_input_tokens, total_output_tokens)

        usage = dict(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            model=model,
            cost=cost,
        )

        return dict(verifiable_data=verifiable_data, usage=usage)

    except Exception as exc:
        raise Exception(f"Error extracting verifiable data: {str(exc)}") from exc
