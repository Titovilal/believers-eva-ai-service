"""
Parsers module for processing documents from various sources.
"""

from .models import ParsedDocument, Chunk, ChunkerProtocol, ParserProtocol
from .chunkers import TextChunker, DoclingNativeChunker, chunk_documents
from .docling_parser import DoclingParser

__all__ = [
    # Data models
    "ParsedDocument",
    "Chunk",
    # Protocols
    "ChunkerProtocol",
    "ParserProtocol",
    # Chunkers
    "TextChunker",
    "DoclingNativeChunker",
    "chunk_documents",
    # Parsers
    "DoclingParser",
]
