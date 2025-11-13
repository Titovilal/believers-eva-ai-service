"""
RAG Module Constants
All default values and configuration constants for the RAG pipeline.
"""

# ==============================================================================
# LANGUAGE DETECTION
# ==============================================================================
LANGUAGE_DEFAULT = "en"


# ==============================================================================
# CHUNKING
# ==============================================================================
CHUNK_DEFAULT_SIZE = 1024
CHUNK_DEFAULT_OVERLAP = 0
CHUNK_DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


# ==============================================================================
# EMBEDDINGS
# ==============================================================================
EMBEDDING_DEFAULT_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL_PRICE = 0.02  # USD per million tokens


# ==============================================================================
# VERIFIABLE DATA EXTRACTION
# ==============================================================================
VERIFIABLE_DEFAULT_EXTRACT = True
VERIFIABLE_DEFAULT_BATCH_SIZE = 5  # Process chunks in batches to reduce API calls

# Model configuration
VERIFIABLE_DEFAULT_MODEL = "gpt-5-mini"
VERIFIABLE_MODEL_INPUT_PRICE = 0.25  # USD per million tokens
VERIFIABLE_MODEL_OUTPUT_PRICE = 2.0  # USD per million tokens


# ==============================================================================
# PDF PARSING - GENERAL
# ==============================================================================
PARSE_PDF_DEFAULT_ENABLE_IMAGE_ANNOTATION = True
PARSE_PDF_DEFAULT_FORCE_OCR = False  # Prefer native text extraction (faster)
PARSE_PDF_DEFAULT_IMAGE_DETAIL = "low"  # "low" or "high" - affects token usage


# ==============================================================================
# PDF PARSING - DOCLING IMAGE ANNOTATION (via OpenAI API)
# ==============================================================================
# API configuration
PARSE_PDF_DOCLING_IMAGE_API_URL = "https://api.openai.com/v1/chat/completions"
PARSE_PDF_DOCLING_IMAGE_TIMEOUT = 34
PARSE_PDF_DOCLING_IMAGE_PROMPT = "Describe the picture."

# Model configuration
PARSE_PDF_DOCLING_IMAGE_MODEL = "gpt-5-nano"
PARSE_PDF_DOCLING_IMAGE_MAX_TOKENS = 2048
PARSE_PDF_DOCLING_IMAGE_AVG_OUTPUT_TOKENS = (
    1024  # Average output tokens per image (half of max 2048)
)

# Token estimation
PARSE_PDF_DOCLING_IMAGE_TOKENS_LOW_DETAIL = 70  # Tokens per image (low detail mode)
PARSE_PDF_DOCLING_IMAGE_TOKENS_HIGH_DETAIL = (
    630  # Estimated average for high detail (4 tiles typical)
)

# Pricing
PARSE_PDF_DOCLING_IMAGE_INPUT_PRICE = 0.05  # USD per million input tokens (vision)
PARSE_PDF_DOCLING_IMAGE_OUTPUT_PRICE = 0.4  # USD per million output tokens (gpt-5-nano)


# ==============================================================================
# PDF PARSING - OPENAI DIRECT OCR (force_ocr=True)
# ==============================================================================
PARSE_PDF_OPENAI_IMAGE_DPI = 200  # DPI for PDF to image conversion

# Model configuration
PARSE_PDF_OPENAI_MODEL = "gpt-5-mini"
PARSE_PDF_OPENAI_INPUT_PRICE = 0.25  # USD per million tokens (estimated)
PARSE_PDF_OPENAI_OUTPUT_PRICE = 2.0  # USD per million tokens (estimated)
