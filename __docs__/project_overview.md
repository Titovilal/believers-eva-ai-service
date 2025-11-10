# Believers EVA AI Service

## What It Does
This is a FastAPI service that provides an AI chat interface using LLM agents and document processing capabilities. It allows users to send messages to AI agents and process documents (PDFs/text) through a REST API, supporting streaming responses, execution flow tracking, and verifiable data extraction via OpenRouter.

### Main Endpoints
- `/chat` - Standard chat endpoint with AI agents
- `/chat/stream` - Streaming chat endpoint with AI agents
- `/process-document` - Document processing endpoint (PDF/text)
- `/health` - Health check endpoint

## Main Files

### API Layer
- `main.py` - Entry point that starts the FastAPI server
- `src/api/routes.py` - Defines API endpoints for chat (standard and streaming), document processing, and health checks
- `src/api/models.py` - Request and response data models with execution flow tracking and document processing
- `src/api/chat_parser.py` - Converts between API formats and pydantic-ai formats

### Agent System
- `src/agents/agent_factory.py` - Factory pattern to create and manage AI agents
- `src/agents/models.py` - Agent configuration and dependencies
- `src/agents/tools.py` - Agent tools (e.g., get_city_temperature)
- `src/agents/promtps.py` - Agent system prompts

### Document Processing
- `src/process_document/process_document.py` - Main document processing pipeline
- `src/process_document/parse_pdf.py` - PDF parsing with Docling
- `src/process_document/generate_chunks.py` - Text chunking with LangChain
- `src/process_document/generate_embeddings.py` - Vector embedding generation
- `src/process_document/detect_number_in_text.py` - Numeric content detection
- `src/process_document/extract_verifiable_data.py` - Verifiable fact extraction

### Utilities
- `src/utils/constants.py` - Configuration constants and defaults
- `src/utils/logs.py` - Logging utilities

## Flow

### Chat Flow
1. User sends a POST request to `/chat` or `/chat/stream` with a message, model name, agent type, and optional history
2. The chat_parser converts the request to pydantic-ai format
3. The AgentFactory creates the appropriate agent with the specified LLM model via OpenRouter
4. The agent runs asynchronously (with or without streaming) and tracks execution flow
5. The response is parsed back to API format with the AI-generated text and execution flow details
6. The API returns the response to the user

### Document Processing Flow
1. User sends a POST request to `/process-document` with base64-encoded document data
2. System detects file type (PDF or text) and parses accordingly
3. Text is chunked into semantic segments with configurable size/overlap
4. Vector embeddings are generated for each chunk
5. Chunks are analyzed for numeric content
6. Verifiable facts with numbers are extracted (optional)
7. Results include text, chunks, embeddings, numeric flags, and verifiable statements
8. Response is returned with all processed data (TODO: upload to database)
