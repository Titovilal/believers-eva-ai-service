"""
Class for ChromaDB retriever that implements the repository pattern.
This wrapper interfaces with ChromaDB for vector-based retrieval with metadata filtering.
It allows for consistent operation with other retriever implementations while leveraging
the vector search capabilities of ChromaDB.
"""

import os
import json
import shutil
import chromadb
from typing import Dict, Any, List, Union, Optional, Tuple
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from dotenv import load_dotenv

from .models import Retriever

# Load environment variables
load_dotenv()


class ChromaDBRetriever(Retriever):
    def __init__(
        self,
        collection_name: str = "default_collection",
        embedding_function_name: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence_transformer",
        persist_directory: Optional[str] = None,
        client_type: str = "local",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the ChromaDBRetriever with optional parameters.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_function_name: Name of the embedding function to use
            embedding_provider: Provider of embeddings ('sentence_transformer', 'openai', 'google', 'huggingface')
            persist_directory: Directory to persist the database (if None, in-memory database is used)
            client_type: Type of ChromaDB client (local or http)
            api_key: Optional API key for embedding models (if not provided, will use environment variables)
        """
        self.collection_name = collection_name
        self.embedding_function_name = embedding_function_name
        self.embedding_provider = embedding_provider
        self.persist_directory = persist_directory
        self.client_type = client_type

        # Initialize embedding function based on provider
        self.embedding_function = self._initialize_embedding_function(
            embedding_provider, embedding_function_name, api_key
        )

        # Initialize client
        if client_type == "local":
            if persist_directory:
                self.client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                self.client = chromadb.Client()
        elif client_type == "http":
            self.client = chromadb.HttpClient()
        else:
            raise ValueError(f"Unsupported client_type: {client_type}")

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

        # Document tracking for consistent interface with other retrievers
        self.document_id_map = {}  # Maps sequential IDs to ChromaDB IDs
        self.documents = []  # Original documents

    def load(self, path: str) -> None:
        """
        Load a previously saved index.

        Args:
            path: Path to the saved index directory
        """
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        # Load document tracking data
        doc_map_path = os.path.join(path, "document_id_map.json")
        docs_path = os.path.join(path, "documents.json")

        if os.path.exists(doc_map_path):
            with open(doc_map_path, "r") as f:
                self.document_id_map = json.load(f)

        if os.path.exists(docs_path):
            with open(docs_path, "r") as f:
                self.documents = json.load(f)

        # Update the persist directory for ChromaDB
        if not self.persist_directory:
            chroma_dir = os.path.join(path, "chroma")
            if os.path.exists(chroma_dir):
                self.persist_directory = chroma_dir
                # Reinitialize the client with the loaded directory
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                # Get the collection
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                )

    def save(self, path: str) -> None:
        """
        Save the index to the given path.

        Args:
            path: Path to save the index
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save document tracking data
        with open(os.path.join(path, "document_id_map.json"), "w") as f:
            json.dump(self.document_id_map, f)

        with open(os.path.join(path, "documents.json"), "w") as f:
            json.dump(self.documents, f)

        # If using in-memory database, create a persistent copy
        if not self.persist_directory:
            chroma_dir = os.path.join(path, "chroma")
            os.makedirs(chroma_dir, exist_ok=True)

            # Create a persistent client and collection
            persistent_client = chromadb.PersistentClient(path=chroma_dir)

            # Check if collection exists in the persistent client
            try:
                persistent_client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                )
                # If it exists, delete it to create a fresh copy
                persistent_client.delete_collection(name=self.collection_name)
            except Exception:
                pass

            # Create a new collection
            persistent_collection = persistent_client.create_collection(
                name=self.collection_name, embedding_function=self.embedding_function
            )

            # Get data from current collection and add to persistent collection
            if self.documents:
                current_data = self.collection.get(
                    include=["embeddings", "metadatas", "documents"]
                )

                if current_data["ids"]:
                    persistent_collection.add(
                        ids=current_data["ids"],
                        embeddings=current_data.get("embeddings"),
                        metadatas=current_data.get("metadatas"),
                        documents=current_data.get("documents"),
                    )

    def build_index(
        self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Build an index from document list with optional metadata.

        Args:
            documents: List of document strings (chunks)
            metadata: Optional list of metadata dictionaries, one per document
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        # Clear existing collection if it exists
        # TODO: THIS IS NOT THE BEST WAY TO HANDLE THIS
        # Check in the code how many create collections calls are made
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception as e:
            print(
                f"Collection '{self.collection_name}' does not exist or could not be deleted: {e}"
            )
        self.collection = self.client.create_collection(
            name=self.collection_name, embedding_function=self.embedding_function
        )

        # Reset document tracking
        self.document_id_map = {}
        self.documents = documents.copy()

        # Generate sequential IDs
        ids = [f"doc_{i}" for i in range(len(documents))]

        # Create document ID map
        self.document_id_map = {i: doc_id for i, doc_id in enumerate(ids)}

        # Add to collection
        self.collection.add(
            ids=ids, documents=documents, metadatas=metadata if metadata else None
        )

    def add_documents(
        self,
        documents: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> None:
        """
        Add one or more documents to the index with optional metadata.
        This method handles both single document and multiple documents.

        Args:
            documents: Either a single document string or a list of document strings
            metadata: Either a single metadata dictionary or a list of metadata dictionaries
                     (must match the number of documents)
        """
        if not documents:
            return

        # Convert single document to list
        if isinstance(documents, str):
            documents = [documents]
            if metadata is isinstance(metadata, list):
                raise ValueError(
                    "If documents is a single string, metadata must also be a single dictionary"
                )
            elif metadata is not None and not isinstance(metadata, list):
                metadata = [metadata]
            elif metadata is None:
                metadata = [{}]
        # Validate metadata size if provided
        if metadata and len(metadata) != len(documents):
            raise ValueError(
                "The size of the metadata list must match the size of the documents list"
            )

        # Generate sequential IDs starting from current count
        start_idx = len(self.documents)
        ids = [f"doc_{i}" for i in range(start_idx, start_idx + len(documents))]

        # Update document tracking
        for i, doc_id in enumerate(ids):
            self.document_id_map[start_idx + i] = doc_id

        self.documents.extend(documents)

        # Add to collection
        self.collection.add(
            ids=ids, documents=documents, metadatas=metadata if metadata else None
        )

    def remove_documents(
        self,
        document_ids: Optional[Union[str, int, List[Union[str, int]]]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Remove documents from the index by IDs and/or metadata filter.

        Args:
            document_ids: Optional ID or list of IDs to remove
            metadata_filter: Optional metadata constraints to filter documents to remove

        Note: At least one of document_ids or metadata_filter must be provided
        """
        if document_ids is None and metadata_filter is None:
            raise ValueError(
                "At least one of document_ids or metadata_filter must be provided"
            )

        # If removing by metadata only
        if document_ids is None and metadata_filter is not None:
            # Get the documents that match the metadata filter
            matching_docs = self.collection.get(where=metadata_filter)
            if not matching_docs["ids"]:
                return  # No documents match the metadata filter

            # Remove the documents
            self.collection.delete(where=metadata_filter)

            # Update document tracking
            chroma_ids_to_remove = matching_docs["ids"]
            seq_ids_to_remove = []

            # Find sequential IDs that correspond to the removed ChromaDB IDs
            for seq_id, chroma_id in list(self.document_id_map.items()):
                if chroma_id in chroma_ids_to_remove:
                    seq_ids_to_remove.append(seq_id)

            # Remove the documents from tracking
            for seq_id in sorted(seq_ids_to_remove, reverse=True):
                self.documents.pop(seq_id)

            # Rebuild document ID map
            new_document_id_map = {}
            i = 0
            for seq_id in sorted(self.document_id_map.keys()):
                if seq_id not in seq_ids_to_remove:
                    chroma_id = self.document_id_map[seq_id]
                    new_document_id_map[i] = chroma_id
                    i += 1

            self.document_id_map = new_document_id_map
            return

        # Process document_ids
        chroma_ids_to_remove = []

        # Convert single ID to list
        if not isinstance(document_ids, list):
            document_ids = [document_ids]

        # Process each document ID
        for doc_id in document_ids:
            # If document_id is an integer (sequential ID), convert to ChromaDB ID
            if isinstance(doc_id, int) or (
                isinstance(doc_id, str) and doc_id.isdigit()
            ):
                idx = int(doc_id)
                if idx not in self.document_id_map:
                    raise ValueError(f"Document ID {idx} not found in index")
                chroma_id = self.document_id_map[idx]
                chroma_ids_to_remove.append(chroma_id)
            else:
                # Assume document_id is already a ChromaDB ID
                chroma_ids_to_remove.append(doc_id)

        # Also filter by metadata if provided
        if metadata_filter is not None:
            self.collection.delete(ids=chroma_ids_to_remove, where=metadata_filter)
        else:
            self.collection.delete(ids=chroma_ids_to_remove)

        # Update document tracking
        seq_ids_to_remove = []
        for seq_id, chroma_id in list(self.document_id_map.items()):
            if chroma_id in chroma_ids_to_remove:
                seq_ids_to_remove.append(seq_id)

        # Remove the documents from tracking
        for seq_id in sorted(seq_ids_to_remove, reverse=True):
            if 0 <= seq_id < len(self.documents):
                self.documents.pop(seq_id)

        # Rebuild document ID map
        new_document_id_map = {}
        i = 0
        for seq_id in sorted(self.document_id_map.keys()):
            if seq_id not in seq_ids_to_remove:
                chroma_id = self.document_id_map[seq_id]
                new_document_id_map[i] = chroma_id
                i += 1

        self.document_id_map = new_document_id_map

    def retrieve(
        self,
        query: Optional[str] = None,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[List[Union[str, int]], List[float], List[str]]:
        """
        Retrieve documents based on query and/or metadata.

        Args:
            query: Query string (optional)
            top_k: Number of top results to return
            metadata_filter: Dictionary with metadata constraints
            threshold: Score threshold for filtering results (lower scores = better in ChromaDB)

        Returns:
            Tuple of (document_indices, scores, document_texts)
        """
        if not query and not metadata_filter:
            raise ValueError("Either query or metadata_filter must be provided")

        # If collection is empty
        if len(self.documents) == 0:
            return [], [], []

        # Handle metadata-only query
        if not query and metadata_filter:
            results = self.collection.get(
                where=metadata_filter, limit=top_k, include=["documents", "metadatas"]
            )

            # Map ChromaDB IDs to sequential indices
            indices = []
            for chroma_id in results["ids"]:
                for idx, doc_id in self.document_id_map.items():
                    if doc_id == chroma_id:
                        indices.append(idx)
                        break

            # Convert distances to similarity scores (1 - normalized distance).
            # Distances in ChromaDB range from 0 (most similar) to 2.
            scores = (
                [1.0] * len(indices)
                if "distances" not in results
                else [1.0 - min(d / 2, 1.0) for d in results["distances"]]
            )
            return indices, scores, results["documents"]

        # Handle query with optional metadata filter
        results = self.collection.query(
            query_texts=[query] if query else None,
            n_results=top_k,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Extract results
        chroma_ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []

        # Convert distances to scores (1 - distance)
        distances = (
            results["distances"][0]
            if "distances" in results and results["distances"]
            else []
        )
        scores = [1.0 - min(d / 2, 1.0) for d in distances]

        # Apply threshold if specified
        if threshold is not None:
            filtered_ids = []
            filtered_scores = []
            filtered_docs = []

            for i, score in enumerate(scores):
                if score >= threshold:
                    filtered_scores.append(score)
                    filtered_docs.append(documents[i])
                    filtered_ids.append(chroma_ids[i])

            chroma_ids = filtered_ids
            documents = filtered_docs
            scores = filtered_scores

        # Map ChromaDB IDs to sequential indices
        indices = []
        for chroma_id in chroma_ids:
            for idx, doc_id in self.document_id_map.items():
                if doc_id == chroma_id:
                    indices.append(idx)
                    break

        return indices, scores, documents

    def destroy_index(self) -> bool:
        """
        Destroy the index and remove all associated files.

        Returns:
            True if successful, False otherwise
        """
        success = True

        try:
            # Delete the collection
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception as e:
                print(f"Error deleting collection: {e}")
                success = False

            # If using a persistent directory, delete it
            if self.persist_directory and os.path.exists(self.persist_directory):
                try:
                    shutil.rmtree(self.persist_directory)
                except OSError as e:
                    print(f"Error removing persist directory: {e}")
                    success = False

            # Reset state
            self.collection = self.client.create_collection(
                name=self.collection_name, embedding_function=self.embedding_function
            )
            self.document_id_map = {}
            self.documents = []

        except Exception as e:
            print(f"Error destroying index: {e}")
            success = False

        return success

    def _initialize_embedding_function(
        self, provider: str, model_name: str, api_key: Optional[str] = None
    ):
        """
        Initialize the embedding function based on the provider.

        Args:
            provider: The embedding provider ('sentence_transformer', 'openai', 'google', 'huggingface')
            model_name: The model name to use
            api_key: Optional API key (if not provided, will use environment variables)

        Returns:
            An embedding function compatible with ChromaDB
        """
        if provider == "sentence_transformer":
            if model_name == "all-MiniLM-L6-v2":
                return embedding_functions.DefaultEmbeddingFunction()
            else:
                return embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
        elif provider == "openai":
            # Get API key from environment if not provided
            openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")

            # Check if using Azure OpenAI
            azure_api_base = os.getenv("AZURE_OPENAI_API_BASE")
            if azure_api_base:
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    api_base=azure_api_base,
                    api_type="azure",
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                    model_name=model_name,
                )
            else:
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name=model_name,
                )
        elif provider == "google":
            # Get API key from environment if not provided
            google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("Google API key is required for Google embeddings")

            return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=google_api_key
            )
        elif provider == "huggingface":
            # Get API key from environment if not provided
            hf_api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
            if not hf_api_key:
                raise ValueError(
                    "HuggingFace API key is required for HuggingFace embeddings"
                )

            return embedding_functions.HuggingFaceEmbeddingFunction(
                api_key=hf_api_key,
                model_name=model_name,
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def count(self) -> int:
        """
        Get the number of documents in the index.

        Returns:
            The count of documents in the collection
        """
        return self.collection.count()
