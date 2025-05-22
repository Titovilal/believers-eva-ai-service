"""
This module defines an abstract class for retrievers following the repository pattern.
All retriever implementations should inherit from this abstract base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional, Tuple


class Retriever(ABC):
    """
    Abstract base class for retriever implementations.
    Implements the repository pattern for a uniform interface across different retriever types.
    """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a previously saved index.

        Args:
            path: Path to the saved index
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the index to the given path.

        Args:
            path: Path to save the index
        """
        pass

    @abstractmethod
    def build_index(
        self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Build an index from document list with optional metadata.

        Args:
            documents: List of document strings (chunks)
            metadata: Optional list of metadata dictionaries, one per document
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
            threshold: Score threshold for filtering results

        Returns:
            Tuple of (document_ids, scores, document_texts)
        """
        pass

    @abstractmethod
    def destroy_index(self) -> bool:
        """
        Destroy the index and remove all associated files.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Get the number of documents in the index.

        Returns:
            The count of documents in the index
        """
        pass
