"""
Routes Module
API route definitions and handlers.
"""

from fastapi import APIRouter
from src.api.models import ChatRequest, ChatResponse
from src.agents.agent_factory import AgentFactory
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint to chat with the AI agent."""

    # Create agent using factory
    agent = AgentFactory.create_agent(
        agent_type=request.agent_type,
        openrouter_model=request.model,
    )

    # Setup the agent
    agent.setup()

    # Run the agent with the user's message
    result = await agent.run_async(request.message)

    return ChatResponse(response=result.data)


@router.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}
