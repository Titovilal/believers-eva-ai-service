"""
Module containing various chunking strategies for document processing.
"""

from typing import List
from .models import ParsedDocument, Chunk, ChunkerProtocol
from docling_core.types import DoclingDocument
from docling.chunking import HybridChunker


class DoclingNativeChunker(ChunkerProtocol):
    """
    Uses the native Docling chunking capabilities.
    """

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        if document.raw_document is None or not isinstance(
            document.raw_document, DoclingDocument
        ):
            raise ValueError("This chunker requires the original Docling document")

        # Implementation using Docling's native chunker
        chunker = HybridChunker()
        chunks = []
        for i, chunk in enumerate(chunker.chunk(dl_doc=document.raw_document)):
            enriched_text = chunker.contextualize(chunk=chunk)
            chunks.append(
                Chunk(text=enriched_text, metadata={**document.metadata, "chunk_id": i})
            )
        return chunks


class TextChunker(ChunkerProtocol):
    """
    A simple text-based chunker that works on the parsed text.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        text = document.text
        chunks = []

        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i : i + self.chunk_size]
            if not chunk_text.strip():  # Skip empty chunks
                continue

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={**document.metadata, "chunk_id": len(chunks)},
                )
            )

        return chunks


def chunk_documents(
    documents: List[ParsedDocument], chunker: ChunkerProtocol
) -> List[Chunk]:
    """
    Utility function to chunk a list of parsed documents using the specified chunker.

    Args:
        documents: List of parsed documents
        chunker: A chunker implementing the ChunkerProtocol

    Returns:
        A list of chunks
    """
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)

    return all_chunks
