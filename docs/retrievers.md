# Retriever System with Repository Pattern

This document provides an overview of the retriever system implemented using the repository pattern, which supports multiple retrieval strategies through a consistent interface.

## Overview

The retriever system is designed around an abstract `Retriever` class that defines a common interface for all retriever implementations. Currently, there are two concrete implementations:

1. **BM25sRetriever** - a keyword-based retrieval system using the BM25 algorithm
2. **ChromaDBRetriever** - a vector-based retrieval system using semantic embeddings

Both implementations provide the same interface for:
- Building and searching indices
- Adding and removing documents
- Metadata filtering
- Persistence (saving/loading)
- Document counting

## Installation

Each retriever has its own dependencies:

### For BM25sRetriever
```bash
pip install bm25s PyStemmer numpy
```

### For ChromaDBRetriever
```bash
pip install chromadb sentence-transformers
```

Optionally, for additional embedding providers with ChromaDBRetriever:
```bash
pip install openai google-generativeai huggingface-hub
```

## Common Usage Pattern

Both retrievers implement the same interface, allowing for interchangeable use:

```python
from src.retrievers import BM25sRetriever, ChromaDBRetriever

# Create a retriever (either type)
retriever = BM25sRetriever()  # or ChromaDBRetriever()

# Sample documents and metadata
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Information retrieval systems find documents based on queries",
    "Python is a high-level programming language"
]

metadata = [
    {"category": "example", "source": "classic_phrase"},
    {"category": "retrieval", "source": "technical"},
    {"category": "programming", "source": "technical"}
]

# Build the index
retriever.build_index(documents, metadata)

# Basic query
indices, scores, docs = retriever.retrieve(query="retrieval")

# Query with metadata filtering
indices, scores, docs = retriever.retrieve(
    query="programming",
    metadata_filter={"category": "programming"}
)

# Metadata-only query
indices, scores, docs = retriever.retrieve(
    metadata_filter={"source": "technical"}
)

# Add documents (single or multiple)
retriever.add_documents(
    "New document about information retrieval",
    {"category": "retrieval", "source": "blog"}
)

# Remove documents by ID or metadata
retriever.remove_documents(document_ids=0)  # Single document
retriever.remove_documents(document_ids=[1, 2])  # Multiple documents
retriever.remove_documents(metadata_filter={"source": "blog"})

# Count documents in the index
doc_count = retriever.count()
print(f"Number of documents: {doc_count}")

# Save and load the index
retriever.save("my_index")
retriever.load("my_index")

# Clean up
retriever.destroy_index()
```

## BM25sRetriever

### Description
BM25sRetriever uses the BM25 algorithm for keyword-based retrieval, which is based on term frequency-inverse document frequency (TF-IDF) scoring with additional parameters to control term saturation and document length normalization.

### Features
- Keyword-based search using the BM25 algorithm
- Language support via stemming and stopwords
- Exact keyword matching with relevance scoring
- Configurable language and stemming options

### Specific Configuration
```python
# Create with language and stemming options
retriever = BM25sRetriever(
    language="english",  # Default
    use_stemmer=True,    # Default
    index_path=None      # Path to load existing index
)
```

### When to Use
BM25sRetriever is best suited for:
- Scenarios where exact keyword matching is important
- Cases where you need to explain why a document was retrieved (keywords)
- Systems with limited computational resources (no embedding models)
- Applications where precise term matching outweighs semantic understanding

## ChromaDBRetriever

### Description
ChromaDBRetriever uses vector embeddings for semantic search, where documents and queries are transformed into high-dimensional vectors, and similarity is computed based on vector proximity.

### Features
- Semantic search using vector embeddings
- Multiple embedding providers:
  - SentenceTransformer (default)
  - OpenAI
  - Google Gemini
  - HuggingFace
- Persistent or in-memory vector database
- Support for different embedding models

### Specific Configuration
```python
# Basic configuration
retriever = ChromaDBRetriever(
    collection_name="my_collection",  # Default: "default_collection"
    embedding_function_name="all-MiniLM-L6-v2",  # Default embedding model
    embedding_provider="sentence_transformer",  # Default provider
    persist_directory=None,  # If provided, enables persistent storage
    client_type="local"  # "local" or "http"
)

# OpenAI embeddings
retriever = ChromaDBRetriever(
    embedding_function_name="text-embedding-3-small",
    embedding_provider="openai"
)

# Google Gemini embeddings
retriever = ChromaDBRetriever(
    embedding_provider="google"
)

# HuggingFace embeddings
retriever = ChromaDBRetriever(
    embedding_function_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_provider="huggingface"
)
```

### When to Use
ChromaDBRetriever is best suited for:
- Semantic search capabilities where meaning matters more than exact keywords
- Finding conceptually similar documents
- Applications where understanding nuance and context is important
- Systems with available computational resources for embedding models

## Comparison

| Feature | BM25sRetriever | ChromaDBRetriever |
|---------|---------------|------------------|
| Search Type | Keyword-based | Semantic (vector-based) |
| Strengths | Precise keyword matching, Fast, Lightweight | Understands meaning, Finds conceptually similar content |
| Weaknesses | Misses semantic relationships, Synonym issues | Computationally intensive, Requires embedding models |
| Storage Requirements | Lower | Higher (stores embeddings) |
| External Dependencies | bm25s, PyStemmer | chromadb, sentence-transformers |
| When to Use | Exact matching, Limited resources, Explainability | Semantic understanding, Conceptual similarity |

## Example Script

Check out the example script in `examples/retrievers_examples.py` for a complete demonstration of both retrievers' capabilities and a direct comparison of their performance on the same dataset.

## Implementation Details

Both retrievers implement the abstract `Retriever` class, which defines the following methods:

- `load(path)` - Load a previously saved index
- `save(path)` - Save the index to a given path
- `build_index(documents, metadata)` - Build an index from documents and metadata
- `add_documents(documents, metadata)` - Add documents to an existing index
- `remove_documents(document_ids, metadata_filter)` - Remove documents by ID or metadata
- `retrieve(query, top_k, metadata_filter, threshold)` - Retrieve documents based on query/metadata
- `destroy_index()` - Clean up the index and associated files
- `count()` - Get the number of documents in the index

This common interface enables consistent usage patterns across different retriever implementations.
