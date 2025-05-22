from src.retrievers import Retriever
from dataclasses import dataclass


@dataclass
class Deps:
    """Dependencies"""

    user_question: str | None = None
    retriever: Retriever | None = None
    inference_time: float = 0.0
