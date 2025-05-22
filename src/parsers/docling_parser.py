"""
Module for parsing documents using the Docling library.

This module provides the DoclingParser class which handles converting documents
to structured format using Docling's parsing capabilities.
"""

from typing import Union, List
from docling.document_converter import DocumentConverter

from .models import ParsedDocument, ParserProtocol, Chunk
from .chunkers import ChunkerProtocol, chunk_documents


class DoclingParser(ParserProtocol):
    """
    This class is responsible for parsing PDFs using the Docling library.
    """

    def __init__(self):
        self._converter = DocumentConverter()

    def parse(self, documents: Union[str, List[str]]) -> List[ParsedDocument]:
        """
        Function that parses a document or a list of documents and returns a list of ParsedDocuments.

        Args:
            documents: A single document path or a list of document paths

        Returns:
            A list of ParsedDocument objects
        """
        if isinstance(documents, str):
            documents = [documents]

        result = []
        for document in documents:
            filename = document.split("/")[-1]
            docling_doc = self._converter.convert(document)
            parsed_doc = docling_doc.document
            md_doc = parsed_doc.export_to_markdown()

            result.append(
                ParsedDocument(
                    filename=filename,
                    text=md_doc,
                    metadata={"file": filename},
                    raw_document=parsed_doc,  # Store the raw docling document
                    parser_type="docling",
                )
            )

        return result

    def chunk_documents(
        self, documents: List[ParsedDocument], chunker: ChunkerProtocol
    ) -> List[Chunk]:
        """
        Chunk the parsed documents using the specified chunker.

        Args:
            documents: List of parsed documents
            chunker: A chunker implementing the ChunkerProtocol

        Returns:
            A list of chunks
        """
        return chunk_documents(documents, chunker)
