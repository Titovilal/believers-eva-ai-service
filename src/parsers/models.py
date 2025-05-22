"""
Module containing shared data models and protocols for document parsing.
"""

from typing import List, Optional, Protocol, Any
from pydantic import BaseModel


class ParsedDocument(BaseModel):
    """
    This class represents a parsed document.
    """

    filename: str
    text: str
    metadata: dict
    # Store the original document for chunkers that need it
    raw_document: Optional[Any] = None
    parser_type: Optional[str] = None


class Chunk(BaseModel):
    """
    A chunk of text extracted from a document.
    """

    text: str
    metadata: dict


class ChunkerProtocol(Protocol):
    """
    Protocol defining the interface for chunkers.
    """

    def chunk(self, document: ParsedDocument) -> List[Chunk]: ...


class ParserProtocol(Protocol):
    """
    Protocol defining the interface for document parsers.
    """

    def parse(self, documents) -> List[ParsedDocument]: ...
