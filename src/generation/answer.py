"""
Generation step: call Groq LLM with retrieved context and return the answer.

Exposes two functions:
  answer()        — blocking, returns full text + sources dict
  stream_answer() — streaming, returns (groq_stream, chunks) for Streamlit
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.prompts import SYSTEM_PROMPT, build_context
from src.retrieval.dense import retrieve_dense
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.reranker import rerank

GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 9
FETCH = 25


def answer(
    query: str,
    model: SentenceTransformer,
    collection,
    bm25: BM25Okapi,
    all_chunks: list[dict],
    reranker: CrossEncoder,
    k: int = TOP_K,
    mode: str = "hybrid_rerank",
) -> dict:
    """
    Full RAG pipeline — returns {"answer": str, "sources": list[dict]}.

    mode="hybrid_rerank" — BM25 + dense + RRF → cross-encoder rerank  (production)
    mode="hybrid"        — BM25 + dense + RRF, no reranker
    mode="baseline"      — dense-only, no BM25, no reranker
    """
    if mode == "baseline":
        chunks = retrieve_dense(query, model, collection, k=k)
    elif mode == "hybrid":
        chunks = retrieve_hybrid(query, model, collection, bm25, all_chunks, k=k, fetch=FETCH)
    else:  # hybrid_rerank (default)
        chunks = retrieve_hybrid(query, model, collection, bm25, all_chunks, k=FETCH, fetch=FETCH)
        chunks = rerank(query, chunks, reranker, top_n=k)

    context = build_context(chunks)
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return {"answer": response.choices[0].message.content, "sources": chunks}


def stream_answer(
    query: str,
    model: SentenceTransformer,
    collection,
    bm25: BM25Okapi,
    all_chunks: list[dict],
    reranker: CrossEncoder,
    k: int = TOP_K,
) -> tuple:
    """Streaming pipeline. Returns (groq_stream, chunks) for Streamlit write_stream."""
    chunks = retrieve_hybrid(query, model, collection, bm25, all_chunks, k=FETCH, fetch=FETCH)
    chunks = rerank(query, chunks, reranker, top_n=k)
    context = build_context(chunks)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )
    return stream, chunks
