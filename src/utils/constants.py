"""
RAG Module Constants
All default values and configuration constants for the RAG pipeline.
"""

# Chunking constants
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]

# Embedding constants
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Verifiable data extraction constants
DEFAULT_EXTRACT_VERIFIABLE = True
DEFAULT_VERIFIABLE_MODEL = "gpt-5-mini"
DEFAULT_VERIFIABLE_TEMPERATURE = 0.1

# Language detection constants
DEFAULT_LANGUAGE = "en"

# PDF parsing and image annotation constants
DEFAULT_ENABLE_IMAGE_ANNOTATION = True
IMAGE_ANNOTATION_MODEL = "gpt-5-nano"
IMAGE_ANNOTATION_MAX_TOKENS = 512
IMAGE_ANNOTATION_PROMPT = (
    "Describe the image in three sentences. Be concise and accurate."
)
IMAGE_ANNOTATION_TIMEOUT = 34
IMAGE_ANNOTATION_API_URL = "https://api.openai.com/v1/chat/completions"
