"""
Constants for the document processing pipeline
"""

PARSE_PDF = {
    "docling": {
        "model_id": "gpt-5-nano",
        "reasoning": "minimal",  # "minimal", "low", "medium", "high" - affects token usage
        "image_detail": "low",  # "low" or "high" - affects token usage
        "timeout": 30,  # 30 sec per image API call
        "concurrency": 8,  # Number of concurrent API calls
        "model_max_tokens": 2048,
    },
    "openai": {
        "model_id": "gpt-5-mini",
        "reasoning": "minimal",  # "minimal", "low", "medium", "high" - affects token usage
        "image_detail": "low",  # "low" or "high" - affects token usage
        "image_dpi": 200,  # DPI for PDF to image conversion
    },
}

DETECT_NUMBERS = {
    "language": "en",
}

VERIFIABLE_DATA = {
    "model_id": "gpt-5-mini",
    "reasoning": "minimal",  # "minimal", "low", "medium", "high" - affects token usage
    "batch_size": 5,
}

CHUNKING = {
    "chunk_size": 1024,
    "chunk_overlap": 0,
    "separators": ["\n\n", "\n", " ", ""],
}

EMBEDDINGS = {
    "model_id": "text-embedding-3-small",
}

# -------------------------------------------------------------------

MODEL_PRICING = {  # All prices are in USD
    "text-embedding-3-small": {
        "input_price": 0.02,
        "output_price": 0.0,  # No output tokens for embeddings
        "pricing_unit": 1_000_000,
    },
    "gpt-5-nano": {
        "input_price": 0.05,
        "output_price": 0.4,
        "pricing_unit": 1_000_000,
    },
    "gpt-5-mini": {
        "input_price": 0.25,
        "output_price": 2.0,
        "pricing_unit": 1_000_000,
    },
}


def calculate_cost(model_id: str, in_tokens: int, out_tokens: int) -> float:
    """Calculate the cost based on model and token usage."""
    pricing = MODEL_PRICING[model_id]
    in_cost = in_tokens * pricing["input_price"]
    out_cost = out_tokens * pricing["output_price"]
    return round((in_cost + out_cost) / pricing["pricing_unit"], 4)
