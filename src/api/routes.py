"""
Routes Module
API route definitions and handlers.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.api.models import ChatRequest, ChatResponse, DocumentRequest, DocumentResponse
from src.api.chat_parser import (
    parse_request_to_pydantic_ai,
    parse_pydantic_ai_to_response,
)
from src.agents.agent_factory import AgentFactory
from src.process_document.process_document import process_document
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()


def _prepare_agent(request: ChatRequest) -> tuple[dict, any]:
    """Prepare and setup agent from request."""
    parsed = parse_request_to_pydantic_ai(request)
    agent = AgentFactory.create_agent(
        agent_type=parsed["agent_type"],
        openrouter_model=parsed["model"],
    )
    agent.setup()
    return parsed, agent


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint to chat with the AI agent."""
    parsed, agent = _prepare_agent(request)

    result = await agent.run_async(
        user_question=parsed["user_question"], message_history=parsed["message_history"]
    )

    return parse_pydantic_ai_to_response(result)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Endpoint to chat with the AI agent with streaming response."""
    parsed, agent = _prepare_agent(request)

    async def stream_generator():
        async with agent.agent.run_stream(
            parsed["user_question"],
            deps=agent.deps,
            message_history=parsed["message_history"],
        ) as result:
            async for chunk in result.stream_text():
                yield chunk

    return StreamingResponse(stream_generator(), media_type="text/plain")


@router.post("/process-document", response_model=DocumentResponse)
async def process_doc(request: DocumentRequest):
    """
    Endpoint to process a document (PDF or text) and extract structured information.
    """
    result = await process_document(
        base64_data=request.base64_data,
        enable_image_annotation=request.enable_image_annotation,
        force_ocr=request.force_ocr,
        lang=request.lang,
    )
    return result


@router.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}
