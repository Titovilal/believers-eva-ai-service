"""
API Models Module
Data models for API interactions.
"""

from pydantic import BaseModel
from typing import Optional, List, Any, Literal, Dict
from src.utils.constants import (
    PARSE_PDF_DEFAULT_ENABLE_IMAGE_ANNOTATION,
    PARSE_PDF_DEFAULT_FORCE_OCR,
    LANGUAGE_DEFAULT,
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
    enable_image_annotation: bool = PARSE_PDF_DEFAULT_ENABLE_IMAGE_ANNOTATION
    force_ocr: bool = PARSE_PDF_DEFAULT_FORCE_OCR
    lang: str = LANGUAGE_DEFAULT


class DocumentResponse(BaseModel):
    """Response model for document processing"""

    text: str
    chunks: List[str]
    embeddings: List[List[float]]
    chunks_with_numbers: List[bool]
    chunk_count: int
    file_type: str
    verifiable_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    page_count: Optional[int] = None
    file_name: Optional[str] = None
