"""
Agent Data Models Module
Data models for different types of agents.
"""

from dataclasses import dataclass


@dataclass
class BaseDeps:
    """Base dependencies class"""

    inference_time: float = 0.0
    context: dict | None = None
