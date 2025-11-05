"""
Extract verifiable data from text chunks using AI analysis.
"""

import os
import json
from typing import List, Dict, Any
from openai import OpenAI

from ..utils.constants import DEFAULT_VERIFIABLE_MODEL, DEFAULT_VERIFIABLE_TEMPERATURE


def extract_verifiable_data(
    chunks: List[str],
    chunks_with_numbers: List[bool],
    model: str = DEFAULT_VERIFIABLE_MODEL,
    temperature: float = DEFAULT_VERIFIABLE_TEMPERATURE,
) -> Dict[str, Any]:
    """
    Analyze chunks containing numbers and extract verifiable data using AI.

    Args:
        chunks: List of text chunks
        chunks_with_numbers: Boolean list indicating which chunks contain numbers
        model: OpenAI chat model to use for analysis (default: "gpt-4o-mini")
        temperature: Model temperature for generation (default: 0.1 for consistency)

    Returns:
        dict: JSON structure containing extracted verifiable data with format:
            {
                "verifiable_facts": [
                    {
                        "chunk_index": int,
                        "statements": ["frase con dato fehaciente 1", "frase con dato fehaciente 2", ...]
                    },
                    ...
                ],
                "summary": {
                    "total_chunks_analyzed": int,
                    "total_statements_extracted": int
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
            "verifiable_facts": [],
            "summary": {"total_chunks_analyzed": 0, "total_statements_extracted": 0},
        }

    try:
        client = OpenAI(api_key=api_key)
        verifiable_data = []

        system_prompt = """Eres un asistente experto en extraer datos fehacientes y verificables de textos.
Tu tarea es analizar fragmentos de texto y extraer frases cortas que contengan información verificable como:
- Fechas, números, cantidades, porcentajes
- Métricas, estadísticas, mediciones
- Afirmaciones con datos concretos

Extrae frases cortas (máximo 15-20 palabras) que contengan estos datos fehacientes.
Cada frase debe ser concisa pero mantener suficiente contexto para entenderse.

Responde ÚNICAMENTE con un objeto JSON válido con esta estructura:
{
    "statements": ["frase con dato fehaciente 1", "frase con dato fehaciente 2", ...]
}

Si no encuentras datos fehacientes, devuelve un array vacío."""

        for chunk_index, chunk_text in chunks_to_analyze:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Analiza el siguiente texto y extrae datos fehacientes:\n\n{chunk_text}",
                        },
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )

                extracted_json = json.loads(response.choices[0].message.content)
                statements = extracted_json.get("statements", [])

                if statements:
                    verifiable_data.append(
                        {
                            "chunk_index": chunk_index,
                            "statements": statements,
                        }
                    )

            except Exception as e:
                # Log error but continue processing other chunks
                verifiable_data.append(
                    {
                        "chunk_index": chunk_index,
                        "statements": [],
                        "error": str(e),
                    }
                )

        # Count total statements extracted
        total_statements = sum(
            len(item.get("statements", []))
            for item in verifiable_data
            if "error" not in item
        )

        return {
            "verifiable_facts": verifiable_data,
            "summary": {
                "total_chunks_analyzed": len(chunks_to_analyze),
                "total_statements_extracted": total_statements,
            },
        }

    except Exception as e:
        raise Exception(f"Error extracting verifiable data: {str(e)}")
