# Parsers Module

This module provides document parsing functionality for the Valorian-ChatbotRAG-DelegadoVirtual project.

## Structure

- `models.py` - Core data models and protocols
- `chunkers.py` - Document chunking strategies
- `docling_parser.py` - Docling-specific document parsing

## Usage

```python
from src.parsers import DoclingParser, TextChunker

# Parse documents
parser = DoclingParser()
parsed_docs = parser.parse("path/to/document.pdf")

# Create chunks
chunker = TextChunker(chunk_size=500, chunk_overlap=100)
chunks = parser.chunk_documents(parsed_docs, chunker)
```

## Extending

### Adding New Parsers

To add a new parser, implement the `ParserProtocol` interface:

```python
from src.parsers.models import ParsedDocument, ParserProtocol

class NewParser(ParserProtocol):
    def parse(self, documents):
        # Implementation
        return [ParsedDocument(...)]
```

### Adding New Chunking Strategies

To add a new chunking strategy, implement the `ChunkerProtocol` interface:

```python
from src.parsers.models import ParsedDocument, Chunk, ChunkerProtocol

class NewChunker(ChunkerProtocol):
    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        # Implementation
        return [Chunk(...)]
```
## TODOs / known issues:
- ParsedDocument stores both the extracted text and the document object = double the size
- Improve metadata handling
- Add to the docling parser and chunker args for config (PdfPipelineOptions)
- Ensure the same embedding for RAG and docling parser (tokenizer https://docling-project.github.io/docling/examples/hybrid_chunking/#configuring-tokenization)
