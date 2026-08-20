"""Single-layer RAG retriever.

Interview task
--------------
Retrieval is currently one ranking pass: embedding similarity search in Chroma.

Turn this into a two-layer retrieve-then-rerank pipeline:

1. Recall (layer 1): pull a larger candidate set from Chroma.
   Raise ``RECALL_K`` (for example from 5 to 20).
2. Rerank (layer 2): score those candidates against the user query
   (cross-encoder, LLM judge, or similar) and keep the top ``FINAL_K``.

Implement the scoring in ``rerank()``. Keep ``retrieve()`` as the public
entry point so the agent and API stay unchanged.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from app.rag.vectorstore import get_vectorstore

# Layer 1: how many neighbors the vector store should return.
# Increase this when you add a reranker so layer 2 has enough candidates.
RECALL_K = 5

# How many documents the agent actually sees.
FINAL_K = 5


def retrieve(
    query: str,
    *,
    brand: str | None = None,
    condition: str | None = None,
) -> list[Document]:
    """Return the top listings for ``query``.

    Today this is a single-layer dense retriever. After the interview
    change, this function should recall ``RECALL_K`` candidates and then
    rerank them down to ``FINAL_K``.
    """
    vectorstore = get_vectorstore()
    search_kwargs: dict[str, Any] = {"k": RECALL_K}
    metadata_filter = _metadata_filter(brand, condition)
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    candidates = vectorstore.similarity_search(query, **search_kwargs)
    ranked = rerank(query, candidates)
    return ranked[:FINAL_K]


def rerank(query: str, documents: list[Document]) -> list[Document]:
    """Second-stage reranker.

    Currently a no-op: documents stay in vector-similarity order.

    Replace this with a real reranker. Typical approach:

    - Encode (query, document.page_content) pairs with a cross-encoder
    - Sort by relevance score, descending
    - Return the sorted list

    ``query`` is the same natural-language string used for recall.
    """
    return documents


def _metadata_filter(brand: str | None, condition: str | None) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if brand:
        clauses.append({"brand": brand})
    if condition:
        clauses.append({"condition": condition})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
