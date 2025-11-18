"""
Constants for the document processing pipeline
"""

PARSE_PDF = {
    "docling": {
        "image_detail": "low",  # "low" or "high" - affects token usage
        "reasoning": "minimal",  # "minimal", "low", "medium", "high" - affects token usage
        "timeout": 30,  # 30 sec per image API call
        "concurrency": 8,  # Number of concurrent API calls
        "model_id": "gpt-5-nano",  # Model used for image description
        "model_input_price": 0.05,  # USD per million input tokens (vision)
        "model_output_price": 0.4,  # USD per million output tokens (gpt-5-nano)
        "model_max_tokens": 2048,
        "model_pricing_unit": 1_000_000,
    },
    "openai": {
        "image_dpi": 200,  # DPI for PDF to image conversion
        "image_detail": "low",  # "low" or "high" - affects token usage
        "reasoning": "minimal",  # "minimal", "low", "medium", "high" - affects token usage
        "model_id": "gpt-5-mini",
        "model_input_price": 0.25,  # USD per million tokens
        "model_output_price": 2.0,  # USD per million tokens
        "model_pricing_unit": 1_000_000,
    },
}

DETECT_NUMBERS = {
    "language": "en",
}

VERIFIABLE_DATA = {
    "batch_size": 5,
    "reasoning": "minimal",  # "minimal", "low", "medium", "high" - affects token usage
    "model_id": "gpt-5-mini",
    "model_input_price": 0.25,  # USD per million tokens
    "model_output_price": 2.0,  # USD per million tokens
    "model_pricing_unit": 1_000_000,
}

CHUNKING = {
    "chunk_size": 1024,
    "chunk_overlap": 0,
    "separators": ["\n\n", "\n", " ", ""],
}

EMBEDDINGS = {
    "model_id": "text-embedding-3-small",
    "model_price": 0.02,  # USD per million tokens
    "model_pricing_unit": 1_000_000,
}
