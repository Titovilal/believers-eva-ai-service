"""
API Clients Singleton
Centralized management of API clients to avoid redundant initializations.
"""

import threading
from typing import Optional
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


class OpenAIClientSingleton:
    """Thread-safe singleton for OpenAI client."""

    _instance: Optional[OpenAI] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_client(cls) -> OpenAI:
        """Get or create the OpenAI client instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = OpenAI()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None


# Convenience functions for easy access
def get_openai_client() -> OpenAI:
    """Get the global OpenAI client instance."""
    return OpenAIClientSingleton.get_client()


class SQLAlchemySessionSingleton:
    """Thread-safe singleton for SQLAlchemy session factory."""

    _session_factory: Optional[sessionmaker] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_session_factory(cls, database_url: str) -> sessionmaker:
        """Get or create the SQLAlchemy session factory (thread-safe)."""
        if cls._session_factory is None:
            with cls._lock:
                if cls._session_factory is None:
                    engine = create_engine(database_url, echo=True)
                    cls._session_factory = sessionmaker(bind=engine)
        return cls._session_factory

    @classmethod
    def get_session(cls, database_url: str) -> Session:
        """Get a new SQLAlchemy session."""
        factory = cls.get_session_factory(database_url)
        return factory()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            cls._session_factory = None


def get_database_session(database_url: str) -> Session:
    """
    Get a SQLAlchemy database session.

    Example:
        database_url = "postgresql://user:password@localhost:5432/dbname"
    """
    return SQLAlchemySessionSingleton.get_session(database_url)
