"""
Agentic RAG tool: unstructured policy document retrieval.

This wraps the hybrid + rerank retrieval pipeline as a callable tool
for an LLM agent.  The agent calls this when the question is about
policy text (Budget speeches, CPF rules, HDB eligibility, etc.).

Status: stub — wire to agent framework (LangChain / LlamaIndex / raw) when ready.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.loader import Retriever
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.reranker import rerank

FETCH = 25
TOP_K = 9


def search_policy_docs(
    query: str,
    retriever: Retriever,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Retrieve the most relevant policy document chunks for `query`.

    Returns a list of chunk dicts with keys: text, source, document, score.
    """
    chunks = retrieve_hybrid(
        query,
        retriever.model,
        retriever.collection,
        retriever.bm25,
        retriever.chunks,
        k=FETCH,
        fetch=FETCH,
    )
    return rerank(query, chunks, retriever.reranker, top_n=top_k)
