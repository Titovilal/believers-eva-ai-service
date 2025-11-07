"""
Chat Parser Module
Functions for parsing and processing chat messages.
"""

from pydantic_ai.result import FinalResult
from src.api.models import ChatRequest, ChatResponse, ExecutionStep


def parse_request_to_pydantic_ai(request: ChatRequest) -> dict:
    """
    Parse API request to pydantic-ai format.

    Args:
        request: The chat request from the API

    Returns:
        A dictionary with user_question and message_history for pydantic-ai
    """
    return {
        "user_question": request.message.strip(),
        "message_history": request.history if request.history else [],
        "agent_type": request.agent_type,
        "model": request.model,
    }


def parse_pydantic_ai_to_response(result: FinalResult) -> ChatResponse:
    """
    Parse pydantic-ai result to API response.

    Args:
        result: The result from pydantic-ai agent

    Returns:
        ChatResponse object for the API
    """
    # Mapping of part types to their step types and content fields
    PART_TYPE_MAP = {
        "UserPromptPart": ("user_prompt", "content", []),
        "ToolCallPart": ("tool_call", "args", ["tool_name", "tool_call_id"]),
        "ToolReturnPart": ("tool_return", "content", ["tool_name", "tool_call_id"]),
        "TextPart": ("text", "content", []),
    }

    execution_flow = []

    # Extract all events chronologically from messages
    for message in result.all_messages():
        if not hasattr(message, "parts"):
            continue

        for part in message.parts:
            part_type_name = type(part).__name__

            if part_type_name not in PART_TYPE_MAP:
                continue

            step_type, content_field, extra_fields = PART_TYPE_MAP[part_type_name]
            timestamp = getattr(part, "timestamp", None)

            step_data = {
                "type": step_type,
                "content": getattr(part, content_field),
                "timestamp": timestamp.isoformat() if timestamp else None,
            }

            # Add extra fields if they exist in the mapping
            for field in extra_fields:
                step_data[field] = getattr(part, field, None)

            execution_flow.append(ExecutionStep(**step_data))

    return ChatResponse(
        response=result.output,
        execution_flow=execution_flow if execution_flow else None,
    )
