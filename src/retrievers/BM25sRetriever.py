"""
Class for BM25s retriever with metadata filtering similar to ChromaDB: https://docs.trychroma.com/docs/querying-collections/query-and-get
The retriever is based on the BM25 algorithm and uses the bm25s library.
It has a metadata filtering feature that allows you to filter the results based on metadata.
For that purpose, it stores document-metadata relations alongside the index.
If the user provides metadata, the retriever will filter the results based on the metadata prior to retrieval.
If there is no metadata, the retriever will return the top_k results based on the BM25 algorithm.
If there is no query but only metadata, the retriever will return the top_k results based on the metadata.

TODO:
- test remove_documents
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
import bm25s
from Stemmer import Stemmer
import shutil

from .models import Retriever


class BM25sRetriever(Retriever):
    def __init__(
        self, index_path=None, language="english", use_stemmer=True, stopwords="en"
    ):
        """
        Initialize the BM25sRetriever with optional parameters.

        Args:
            index_path: Path to the saved index file (if loading an existing index)
            language: Language for stemming
            use_stemmer: Whether to use stemming
            stopwords: Language code for stopwords or a list of stopwords
        """
        self.language = language
        self.use_stemmer = use_stemmer
        self.stopwords = stopwords
        self.index_path = index_path
        self.stemmer = Stemmer(language) if use_stemmer else None
        self.document_metadata = {}  # Mapping from document index to metadata
        self.documents = []  # Original documents

        if index_path and os.path.exists(index_path):
            self.load(index_path)
        else:
            self.retriever = bm25s.BM25()

    def load(self, path: str) -> None:
        """
        Load a previously saved index and metadata.

        Args:
            path: Path to the saved index
        """
        self.retriever = bm25s.BM25.load(path, mmap=True)

        # Load metadata if it exists
        metadata_path = f"{path}/documents_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.document_metadata = json.load(f)

        # Load original documents if they exist
        docs_path = f"{path}_documents.pkl"
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)

        self.index_path = path

    def build_index(
        self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Build index from document list with optional metadata.

        Args:
            documents: List of document strings (chunks)
            metadata: Optional list of metadata dictionaries, one per document
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        # Store original documents
        self.documents = documents

        # Process metadata
        if metadata:
            if len(metadata) != len(documents):
                raise ValueError(
                    "Metadata list length must match documents list length"
                )
            self.document_metadata = {i: meta for i, meta in enumerate(metadata)}

        # Tokenize and index documents
        corpus_tokens = bm25s.tokenize(
            documents, stopwords=self.stopwords, stemmer=self.stemmer
        )
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def save(self, path: str) -> None:
        """
        Save the index, metadata, and original documents.

        Args:
            path: Path to save the index
        """
        # Save BM25 index
        self.retriever.save(path)

        # Save metadata
        with open(f"{path}/documents_metadata.json", "w") as f:
            json.dump(self.document_metadata, f)

        # Save original documents
        with open(f"{path}_documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        self.index_path = path

    def add_documents(
        self,
        documents: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> None:
        """
        Add one or more documents to the index with optional metadata.
        The bm25s library doesn't support adding documents after indexing.
        This method rebuilds the index with the new documents.

        Args:
            documents: Either a single document string or a list of document strings
            metadata: Either a single metadata dictionary or a list of metadata dictionaries
                     (must match the number of documents)
        """
        # Convert single document to list
        # TODO: test this
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

        # Get existing documents and metadata
        all_documents = self.documents.copy() if self.documents else []
        all_metadata = None

        if self.document_metadata:
            all_metadata = [
                self.document_metadata[i] if i in self.document_metadata else {}
                for i in range(len(all_documents))
            ]

        # Add new documents
        all_documents.extend(documents)

        # Process new metadata
        if metadata:
            if not all_metadata:
                all_metadata = [{} for _ in range(len(all_documents) - len(documents))]

            if len(metadata) != len(documents):
                raise ValueError(
                    "Metadata list length must match new documents list length"
                )

            all_metadata.extend(metadata)

        # Rebuild the index
        self.build_index(all_documents, all_metadata)

    def remove_documents(
        self,
        document_ids: Optional[Union[str, int, List[Union[str, int]]]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Remove documents from the index by IDs and/or metadata filter.
        The bm25s library doesn't support removing documents after indexing.
        This method rebuilds the index without the specified documents.

        Args:
            document_ids: Optional ID or list of IDs to remove
            metadata_filter: Optional metadata constraints to filter documents to remove
        """
        if document_ids is None and metadata_filter is None:
            raise ValueError(
                "At least one of document_ids or metadata_filter must be provided"
            )

        # Convert single ID to list
        if document_ids is not None and not isinstance(document_ids, list):
            document_ids = [document_ids]

        # Convert string IDs to integers
        if document_ids is not None:
            doc_ids_int = []
            for doc_id in document_ids:
                try:
                    doc_ids_int.append(int(doc_id))
                except ValueError:
                    raise ValueError(
                        f"document_id {doc_id} must be an integer or convertible to an integer"
                    )
            document_ids = doc_ids_int

        # Get documents to remove by metadata filter
        if metadata_filter is not None:
            filtered_mask = self._filter_with_metadata(metadata_filter)
            if filtered_mask is not None:
                # Get indices where the mask is True
                filtered_ids = np.where(filtered_mask)[0].tolist()

                # Add to document_ids list if there are matches
                if filtered_ids:
                    if document_ids is None:
                        document_ids = filtered_ids
                    else:
                        document_ids.extend(filtered_ids)
                        # Remove duplicates
                        document_ids = list(set(document_ids))

        # If no documents to remove
        if not document_ids:
            return

        # Create new document and metadata lists without the removed documents
        new_documents = []
        new_metadata = []

        for i in range(len(self.documents)):
            if i not in document_ids:
                new_documents.append(self.documents[i])
                if self.document_metadata and i in self.document_metadata:
                    new_metadata.append(self.document_metadata.get(i, {}))
                else:
                    new_metadata.append({})

        # Rebuild the index
        if new_documents:
            self.build_index(new_documents, new_metadata)
        else:
            # If all documents are removed, reset to empty state
            self.documents = []
            self.document_metadata = {}
            self.retriever = bm25s.BM25()

    def _filter_with_metadata(self, metadata_filter: Dict[str, Any]):
        """
        Filter documents based on metadata.

        Args:
            metadata_filter: Dictionary with metadata constraints

        Returns:
            A boolean mask array for documents matching the filter
        """
        if not self.document_metadata:
            return None

        # Initialize mask with all True values
        mask = np.ones(len(self.documents), dtype=bool)

        # For each filter key and value
        for key, value in metadata_filter.items():
            for idx in range(len(self.documents)):
                # Skip if document already excluded
                if not mask[idx]:
                    continue

                # Skip if document has no metadata
                if idx not in self.document_metadata:
                    mask[idx] = False
                    continue

                doc_metadata = self.document_metadata[idx]

                # Skip if key not in document metadata
                if key not in doc_metadata:
                    mask[idx] = False
                    continue

                # Check for exact match
                if doc_metadata[key] != value:
                    mask[idx] = False

        return mask if np.any(mask) else None

    def retrieve(
        self,
        query: Optional[str] = None,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[List[int], List[float], List[str]]:
        """
        Retrieve documents based on query and/or metadata.

        Args:
            query: Query string (optional)
            top_k: Number of top results to return
            metadata_filter: Dictionary with metadata constraints
            threshold: Score threshold for filtering results

        Returns:
            Tuple of (document_indices, scores, document_texts)
        """
        if not query and not metadata_filter:
            raise ValueError("Either query or metadata_filter must be provided")

        # If no index has been built yet
        if not hasattr(self, "retriever") or not self.documents:
            return [], [], []

        # Get weight mask from metadata filter
        weight_mask = None
        if metadata_filter:
            weight_mask = self._filter_with_metadata(metadata_filter)

            # If filter excludes all documents, return empty results
            if weight_mask is not None and not np.any(weight_mask):
                return [], [], []

        # If only metadata filter is provided with no query
        if metadata_filter and not query:
            # Get indices where weight_mask is True
            if weight_mask is not None:
                indices = np.where(weight_mask)[0][:top_k].tolist()
                # Set equal scores for all results (1.0)
                scores = [1.0] * len(indices)
                # Get the corresponding documents
                documents = [self.documents[int(idx)] for idx in indices]
                return indices, scores, documents
            return [], [], []

        # Process query
        query_tokens = bm25s.tokenize(
            query, stopwords=self.stopwords, stemmer=self.stemmer
        )

        # Get results - bm25s returns arrays of shape (n_queries, k)
        indices_array, scores_array = self.retriever.retrieve(
            query_tokens, k=top_k, weight_mask=weight_mask
        )

        # Convert to lists for easier processing
        # The first dimension is the query (we only have one query)
        indices = (
            indices_array[0].tolist()
            if indices_array.ndim > 1
            else indices_array.tolist()
        )
        scores = (
            scores_array[0].tolist() if scores_array.ndim > 1 else scores_array.tolist()
        )

        # Apply threshold filtering if specified
        if threshold is not None:
            filtered_indices = []
            filtered_scores = []

            for idx, score in zip(indices, scores):
                if score >= threshold:
                    filtered_indices.append(idx)
                    filtered_scores.append(score)

            indices = filtered_indices
            scores = filtered_scores

        # Get the corresponding documents
        documents = [self.documents[int(idx)] for idx in indices]

        return indices, scores, documents

    def destroy_index(self) -> bool:
        """
        Destroy the index and remove all associated files.
        This is useful for cleanup after using the retriever.

        Returns:
            True if successful, False otherwise
        """
        success = True
        if self.index_path:
            # Remove index file if it exists
            if os.path.exists(self.index_path):
                try:
                    shutil.rmtree(self.index_path)
                except OSError as e:
                    print(f"Error removing index file: {e}")
                    success = False

            # Remove metadata file if it exists
            metadata_path = f"{self.index_path}_metadata.json"
            if os.path.exists(metadata_path):
                try:
                    os.remove(metadata_path)
                except OSError as e:
                    print(f"Error removing metadata file: {e}")
                    success = False

            # Remove documents file if it exists
            docs_path = f"{self.index_path}_documents.pkl"
            if os.path.exists(docs_path):
                try:
                    os.remove(docs_path)
                except OSError as e:
                    print(f"Error removing documents file: {e}")
                    success = False

            # Reset internal state
            self.index_path = None
            # Reset the retriever
            self.retriever = bm25s.BM25()
            self.document_metadata = {}
            self.documents = []

        return success

    def count(self) -> int:
        """
        Get the number of documents in the index.

        Returns:
            The count of documents in the index
        """
        return len(self.documents)
