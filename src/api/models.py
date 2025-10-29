"""
API Models Module
Data models for API interactions.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str = "Hello!"
    model: str = "google/gemini-2.5-flash"
    agent_type: str = "simple"


class ChatResponse(BaseModel):
    response: str
