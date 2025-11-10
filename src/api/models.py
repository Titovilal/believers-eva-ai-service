"""
API Models Module
Data models for API interactions.
"""

from pydantic import BaseModel
from typing import Optional, List, Any, Literal, Dict
from src.utils.constants import (
    DEFAULT_ENABLE_IMAGE_ANNOTATION,
    DEFAULT_FORCE_OCR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_EXTRACT_VERIFIABLE,
    DEFAULT_VERIFIABLE_MODEL,
)


class ExecutionStep(BaseModel):
    """Represents a single step in the execution flow"""

    type: Literal["user_prompt", "tool_call", "tool_return", "text"]
    content: Any
    timestamp: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = "Hello!"
    model: str = "google/gemini-2.5-flash"
    agent_type: str = "simple"
    history: Optional[List[ExecutionStep]] = None


class ChatResponse(BaseModel):
    response: str
    execution_flow: Optional[List[ExecutionStep]] = None


class DocumentRequest(BaseModel):
    """Request model for document processing"""

    base64_data: str
    enable_image_annotation: bool = DEFAULT_ENABLE_IMAGE_ANNOTATION
    force_ocr: bool = DEFAULT_FORCE_OCR
    lang: str = DEFAULT_LANGUAGE


class DocumentResponse(BaseModel):
    """Response model for document processing"""

    text: str
    chunks: List[str]
    embeddings: List[List[float]]
    chunks_with_numbers: List[bool]
    chunk_count: int
    file_type: str
    verifiable_facts: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    page_count: Optional[int] = None
    file_name: Optional[str] = None
