"""
File containing the tools for the agent.
"""

import logging
from typing_extensions import Annotated
from pydantic_ai import RunContext
from .models import Deps

# Configure logger for tools
logger = logging.getLogger(__name__)


async def retrieve_from_documents(
    ctx: RunContext[Deps],
    query: Annotated[
        str,
        "A detailed query string for searching within PDF content. The search must be clear and concise to enable effective matching of relevant text chunks.",
    ],
    top_k: Annotated[
        int,
        "A positive integer specifying the number of top ranked results to retrieve from the document collection.",
    ],
) -> str:
    """
    Retrieves and returns relevant text chunks from PDF documents based on the input query.

    The function tokenizes the query and then leverages the retriever in the dependencies
    to fetch the most relevant document chunks. It then formats and returns these results.

    Args:
        query (str): A detailed query string used for retrieving relevant PDF content.
        top_k (int): A positive integer specifying the number of top ranked results to retrieve.

    Returns:
        str: A concatenated string containing each retrieved chunk annotated with its rank and score.
    """
    logger.info(f"Tool called: retrieve_from_documents with query: {query}")
    assert isinstance(top_k, int), "top_k must be an integer"
    assert top_k > 0, "top_k must be a positive integer"
    assert ctx.deps.retriever.count() > 0, (
        "No documents found in the index. Please check the index path."
    )
    indices, scores, docs = ctx.deps.retriever.retrieve(
        query=query,
        top_k=top_k,
        # metadata_filter=,
    )
    logger.info(f"Retrieved {len(docs)} documents with scores: {scores}")
    result = ""
    for i, (idx, score, doc) in enumerate(zip(indices, scores, docs)):
        result += f"Retrieved chunk with RANK {i + 1} (score: {score:.2f}): {doc}\n\n"

    return result
