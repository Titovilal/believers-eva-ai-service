#!/usr/bin/env python
"""
Basic example showing how to use the parsers module to parse a PDF document
and create text chunks.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import DoclingParser, TextChunker, DoclingNativeChunker


def main():
    """Simple demonstration of using the parsers module."""
    # Define the document path
    document_path = "data/2023-9282.pdf"

    # Check if the document exists
    if not os.path.exists(document_path):
        print(f"Error: Document not found at path '{document_path}'")
        sys.exit(1)

    print(f"Processing document: {document_path}")

    # Step 1: Parse the document using DoclingParser
    parser = DoclingParser()
    parsed_docs = parser.parse(document_path)

    if not parsed_docs:
        print("Error: No documents were parsed")
        sys.exit(1)

    doc = parsed_docs[0]
    print(f"Document parsed: {doc.filename}")
    print(f"Text length: {len(doc.text)} characters")

    # Step 2: Create chunks using TextChunker
    text_chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    text_chunks = parser.chunk_documents(parsed_docs, text_chunker)

    print(f"Created {len(text_chunks)} chunks with TextChunker")

    # Print the first chunk from TextChunker
    if text_chunks:
        print("\nTextChunker - First chunk preview:")
        print(text_chunks[0].text[:300].replace("\n", " ") + "...")

    # Step 3: Create chunks using DoclingNativeChunker
    docling_chunker = DoclingNativeChunker()
    try:
        docling_chunks = parser.chunk_documents(parsed_docs, docling_chunker)
        print(f"\nCreated {len(docling_chunks)} chunks with DoclingNativeChunker")

        # Print the first chunk from DoclingNativeChunker
        if docling_chunks:
            print("\nDoclingNativeChunker - First chunk preview:")
            print(docling_chunks[0].text[:300].replace("\n", " ") + "...")

            # Step 4: Compare the chunking strategies
            print("\n--- Chunking Strategy Comparison ---")
            print(f"TextChunker: {len(text_chunks)} chunks")
            print(f"DoclingNativeChunker: {len(docling_chunks)} chunks")

            # Calculate average chunk sizes
            text_avg_size = (
                sum(len(chunk.text) for chunk in text_chunks) / len(text_chunks)
                if text_chunks
                else 0
            )
            docling_avg_size = (
                sum(len(chunk.text) for chunk in docling_chunks) / len(docling_chunks)
                if docling_chunks
                else 0
            )

            print(f"TextChunker average chunk size: {text_avg_size:.2f} characters")
            print(
                f"DoclingNativeChunker average chunk size: {docling_avg_size:.2f} characters"
            )
    except Exception as e:
        print(f"\nError using DoclingNativeChunker: {str(e)}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
