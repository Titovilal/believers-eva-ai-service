# Believers EVA AI Service

## What It Does
This is a FastAPI service that provides an AI chat interface using LLM agents. It allows users to send messages and receive AI-generated responses through a REST API, supporting multiple agent types and language models via OpenRouter.

## Main Files
- `main.py` - Entry point that starts the FastAPI server
- `src/api/routes.py` - Defines API endpoints for chat and health checks
- `src/api/models.py` - Request and response data models
- `src/agents/agent_factory.py` - Factory pattern to create and manage AI agents
- `src/agents/models.py` - Agent configuration and dependencies

## Flow
1. User sends a POST request to `/chat` with a message, model name, and agent type
2. The AgentFactory creates the appropriate agent with the specified LLM model
3. The agent processes the message using the AI model and returns the response
4. The API returns the AI-generated response to the user
