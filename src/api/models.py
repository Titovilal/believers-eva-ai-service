"""
API Models Module
Data models for API interactions.
"""

from pydantic import BaseModel
from typing import Optional, List, Any, Literal


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
