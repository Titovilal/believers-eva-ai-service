"""
Example script demonstrating how to use the Retriever classes.
This script creates both BM25s and ChromaDB retrievers, indexes documents with metadata,
and shows how to perform queries with metadata filtering for each implementation.
"""

import sys
import os
import shutil

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the retriever classes
from src.retrievers import BM25sRetriever, ChromaDBRetriever


def print_results(indices, scores, docs):
    """Helper function to print retrieval results."""
    print(f"Found {len(indices)} results:")
    for i, (idx, score, doc) in enumerate(zip(indices, scores, docs)):
        # Truncate long documents for display
        doc_display = doc if len(doc) < 60 else doc[:57] + "..."
        print(f"  {i + 1}. [ID: {idx}, Score: {score:.4f}] {doc_display}")


def run_examples(retriever, retriever_name, documents, metadata, index_path):
    """Run a standard set of examples on the given retriever."""
    print(f"\n{'=' * 20} TESTING {retriever_name} {'=' * 20}\n")

    # Build the index
    print(f"Building {retriever_name} index...")
    retriever.build_index(documents, metadata)

    # Save the index
    print(f"Saving index to {index_path}...")
    retriever.save(index_path)

    # Print document count
    print(f"\nNumber of documents in the index: {retriever.count()}")

    print("\n=== Query Examples ===")

    # Example 1: Basic query
    print("\nExample 1: Basic query for 'information retrieval'")
    indices, scores, docs = retriever.retrieve(query="information retrieval", top_k=3)
    print_results(indices, scores, docs)

    # Example 2: Query with metadata filtering
    print(
        "\nExample 2: Query for 'document' with metadata filter {'category': 'retrieval'}"
    )
    indices, scores, docs = retriever.retrieve(
        query="document",
        top_k=3,
        metadata_filter={"category": "retrieval"},
    )
    print_results(indices, scores, docs)

    # Example 3: Query with score threshold
    # Use a fixed threshold for consistency across retrievers
    threshold = 0.3
    print(f"\nExample 3: Query for 'search' with score threshold of {threshold}")
    indices, scores, docs = retriever.retrieve(
        query="search", top_k=5, threshold=threshold
    )
    print_results(indices, scores, docs)

    # Example 4: Metadata-only query
    print("\nExample 4: Metadata-only query with filter {'source': 'definition'}")
    indices, scores, docs = retriever.retrieve(
        metadata_filter={"source": "definition"}, top_k=3
    )
    print_results(indices, scores, docs)

    # Example 5: Add a new document
    print("\nExample 5: Adding a new document to the index")
    new_doc = f"{retriever_name} is a widely used information retrieval algorithm"
    new_meta = {"category": "retrieval", "source": "definition"}
    print(f"\nNumber of documents in the index before addition: {retriever.count()}")
    print(new_doc, new_meta)
    retriever.add_documents(new_doc, new_meta)
    print(f"\nNumber of documents in the index after addition: {retriever.count()}")

    # Query for the new document
    print("\nQuerying for the newly added document:")
    indices, scores, docs = retriever.retrieve(query="widely used algorithm", top_k=1)
    print_results(indices, scores, docs)

    # Example 6: Remove documents by metadata
    print("\nExample 6: Removing documents by metadata filter")
    print(f"\nNumber of documents in the index before removal: {retriever.count()}")
    print("Before removal - documents with source=definition:")
    indices, scores, docs = retriever.retrieve(
        metadata_filter={"source": "definition"}, top_k=5
    )
    print_results(indices, scores, docs)

    # Remove documents with source=definition
    retriever.remove_documents(metadata_filter={"source": "definition"})
    print(f"\nNumber of documents in the index after removal: {retriever.count()}")
    print("\nAfter removal - documents with source=definition:")
    indices, scores, docs = retriever.retrieve(
        metadata_filter={"source": "definition"}, top_k=5
    )
    print_results(indices, scores, docs)

    # Example 7: Remove documents by ID
    print("\nExample 7: Removing a document by ID")
    print(f"\nNumber of documents in the index before removal: {retriever.count()}")
    # Remove document with ID 0
    retriever.remove_documents(document_ids=0)
    print(f"\nNumber of documents in the index after removal: {retriever.count()}")

    # Verify removal
    print("\nVerifying document removal (searching for 'fox'):")
    indices, scores, docs = retriever.retrieve(query="fox", top_k=1)
    print_results(indices, scores, docs)

    print(f"\nDone with {retriever_name} examples!")

    print("\nCleaning up - destroying index...")
    retriever.destroy_index()

    # Clean up saved index
    if os.path.exists(index_path):
        if os.path.isdir(index_path):
            shutil.rmtree(index_path)
        else:
            os.remove(index_path)
        if os.path.exists(f"{index_path}_metadata.json"):  # For BM25s
            os.remove(f"{index_path}_metadata.json")
        print(f"Removed index: {index_path}")


def get_test_data():
    """Return sample documents and metadata suitable for testing all retriever types."""
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Information retrieval systems find relevant documents based on user queries",
        "Python is a high-level programming language widely used in data science",
        "Metadata filtering allows for more precise document retrieval",
        "Semantic search finds documents based on meaning rather than exact keyword matches",
        "Machine learning techniques can improve search results",
        "Natural language processing helps computers understand human language",
        "Vector space models represent documents as vectors in high-dimensional space",
        "Inverted indices allow for efficient keyword lookups in document collections",
        "Embedding models transform text into numerical vectors that capture meaning",
    ]

    metadata = [
        {"category": "example", "source": "classic_phrase"},
        {"category": "retrieval", "source": "technical"},
        {"category": "programming", "source": "technical"},
        {"category": "retrieval", "source": "technical"},
        {"category": "retrieval", "source": "definition"},
        {"category": "machine_learning", "source": "technical"},
        {"category": "nlp", "source": "definition"},
        {"category": "retrieval", "source": "technical"},
        {"category": "retrieval", "source": "technical"},
        {"category": "embedding", "source": "definition"},
    ]

    return documents, metadata


def main():
    """Run examples for both BM25s and ChromaDB retrievers."""
    # Choose which retrievers to test (set to True to test)
    test_bm25s = True
    test_chromadb = True

    # Get common test data for all retrievers
    documents, metadata = get_test_data()
    print(f"Using {len(documents)} test documents for all retrievers")

    # BM25s retriever
    if test_bm25s:
        print("\nInitializing BM25sRetriever...")
        bm25s_retriever = BM25sRetriever(language="english", use_stemmer=True)
        run_examples(
            bm25s_retriever, "BM25s", documents, metadata, "example_bm25s_index"
        )

    # ChromaDB retriever
    if test_chromadb:
        print("\nInitializing ChromaDBRetriever...")
        chromadb_retriever = ChromaDBRetriever(
            collection_name="example_collection",
            embedding_function_name="all-MiniLM-L6-v2",
            embedding_provider="sentence_transformer",
        )
        run_examples(
            chromadb_retriever,
            "ChromaDB",
            documents,
            metadata,
            "example_chromadb_index",
        )

    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
