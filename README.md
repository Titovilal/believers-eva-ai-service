# Believers EVA AI Service

FastAPI service providing AI chat capabilities and document processing with LLM agents through OpenRouter integration.

## Features

- **AI Chat Interface**: REST API for conversational AI using multiple LLM models
- **Document Processing**: PDF and text document parsing with intelligent chunking
- **Verifiable Data Extraction**: Automatic extraction of verifiable facts and numeric statements
- **Embedding Generation**: Vector embeddings for semantic search and RAG applications
- **Agent System**: Extensible agent factory pattern for different AI behaviors
- **OpenRouter Integration**: Support for multiple LLM providers (Gemini, GPT, Claude, etc.)

## Project Structure

```
believers-eva-ai-service/
|-- main.py                   # FastAPI application entry point
|-- pyproject.toml            # Python dependencies
|-- src/
|   |-- api/                  # API layer
|   |   |-- routes.py         # Endpoint definitions
|   |   |-- models.py         # Request/response models
|   |   `-- chat_parser.py    # Chat parsing utilities
|   |-- agents/               # AI agent system
|   |   |-- agent_factory.py  # Agent creation and management
|   |   |-- models.py         # Agent data models
|   |   |-- tools.py          # Agent tools
|   |   `-- promtps.py        # Agent prompts
|   |-- process_document/     # Document processing pipeline
|   |   |-- process_document.py        # Main processing logic
|   |   |-- parse_pdf.py               # PDF parsing with Docling
|   |   |-- generate_chunks.py         # Text chunking
|   |   |-- generate_embeddings.py     # Vector embeddings
|   |   |-- detect_number_in_text.py   # Numeric detection
|   |   `-- extract_verifiable_data.py # Fact extraction
|   `-- utils/                # Shared utilities
|       |-- constants.py      # Configuration constants
|       `-- logs.py           # Logging utilities
|-- cookbooks/                # Example scripts and demos
|   |-- base_cookbook.py      # Base cookbook class
|   |-- execute_*.py          # Individual feature examples
|   `-- run_all_cookbooks.py  # Run all examples
|-- data/
|   `-- cookbooks_input/      # Sample data for testing
`-- __docs__/                 # Project documentation
```

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd believers-eva-ai-service
   ```

2. **Install dependencies**

   Using uv (recommended):

   ```bash
   uv sync
   ```

   Or using pip:

   ```bash
   pip install -e .
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your credentials:

   ```env
   OPENROUTER_API_KEY='your_openrouter_api_key'
   DATABASE_URL_NEON='your_neon_database_url'
   ```

## Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: API key for OpenRouter (required for AI features)
- `DATABASE_URL_NEON`: Neon PostgreSQL connection string (if using database features)

### Default Settings

Default configuration values are in `src/utils/constants.py`:

- **Chunking**: 512 characters per chunk, 0 overlap
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Verifiable extraction**: GPT-5-mini model
- **Language**: Spanish (es) for number detection
- **Image annotation**: Disabled by default

## Usage

### Starting the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### POST /chat

Send a message to an AI agent and receive a response.

**Request:**

```json
{
  "message": "Hello, how are you?",
  "model": "google/gemini-2.5-flash",
  "agent_type": "simple"
}
```

**Response:**

```json
{
  "response": "I'm doing well, thank you for asking! How can I help you today?"
}
```

**Parameters:**

- `message` (string): User's message to the AI
- `model` (string): OpenRouter model identifier (e.g., "google/gemini-2.5-flash", "openai/gpt-4")
- `agent_type` (string): Type of agent to use (currently supports "simple")

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "ok"
}
```

## Document Processing

The document processing module supports:

### Supported Formats

- PDF files (with optional image annotation)
- Plain text files

### Processing Pipeline

1. **Parse**: Extract text from PDF or decode text file
2. **Chunk**: Split into semantic chunks with configurable size/overlap
3. **Embed**: Generate vector embeddings using OpenAI models
4. **Detect Numbers**: Identify chunks containing numeric data
5. **Extract Facts**: Extract verifiable statements with numbers (optional)

### Example Usage

```python
from src.process_document import process_document
import base64

# Read and encode document
with open("document.pdf", "rb") as f:
    base64_data = base64.b64encode(f.read()).decode()

# Process document
result = process_document(
    base64_data=base64_data,
    chunk_size=512,
    chunk_overlap=50,
    extract_verifiable=True,
    lang="es"
)

# Access results
print(f"Extracted {len(result['chunks'])} chunks")
print(f"Generated {len(result['embeddings'])} embeddings")
print(f"Found {sum(result['chunks_with_numbers'])} chunks with numbers")
if 'verifiable_facts' in result:
    print(f"Extracted {result['verifiable_facts']['summary']['total_statements_extracted']} verifiable statements")
```

## Development

### Running Examples

Execute individual cookbooks:

```bash
python cookbooks/execute_process_document.py
python cookbooks/execute_generate_embeddings.py
python cookbooks/execute_extract_verifiable_data.py
```

Or run all cookbooks:

```bash
python cookbooks/run_all_cookbooks.py
```

### Code Style

This project uses:

- **Ruff**: For linting and formatting
- **Python 3.13+**: Type hints throughout

### Project Dependencies

Key libraries:

- `fastapi`: Web framework
- `pydantic-ai`: AI agent framework
- `docling`: Advanced PDF parsing
- `langchain-text-splitters`: Text chunking
- `tiktoken`: Token counting
- `pypdf`: PDF processing
- `text2num`: Number detection in text

## Architecture

### Agent System

The agent system uses a factory pattern:

1. **BaseAgent**: Abstract base class defining agent interface
2. **SimpleAgent**: Basic conversational agent
3. **AgentFactory**: Creates agents based on type and configuration

Agents use OpenRouter for model access, supporting providers like:

- Google (Gemini)
- OpenAI (GPT)
- Anthropic (Claude)
- Meta (Llama)
- And many more

### Document Processing Flow

```
Base64 Input -> Decode -> Parse (PDF/Text) -> Generate Chunks ->
Generate Embeddings -> Detect Numbers -> Extract Verifiable Data -> Return Results
```

### API Flow

```
Client Request -> FastAPI Router -> Agent Factory -> Create Agent ->
Setup Agent -> Run Agent (OpenRouter) -> Return Response
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For questions or issues, please [create an issue](link-to-issues) or contact the maintainers.
