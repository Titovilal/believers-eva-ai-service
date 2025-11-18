"""
API Clients Singleton
Centralized management of API clients to avoid redundant initializations.
"""

import threading
from typing import Optional
from openai import OpenAI


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


# Convenience function for easy access
def get_openai_client() -> OpenAI:
    """Get the global OpenAI client instance."""
    return OpenAIClientSingleton.get_client()
