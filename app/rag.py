"""
RAG pipeline: embed query → hybrid retrieve (BM25 + dense + RRF) → rerank → generate.

Pipeline stages:
  1. Hybrid retrieval — BM25 (keyword) + dense (vector) each return top-FETCH candidates.
  2. RRF fusion      — Reciprocal Rank Fusion merges both ranked lists without score normalisation.
  3. Cross-encoder rerank — re-scores fused candidates; returns top-TOP_K to the LLM.
  4. Generation      — Groq Llama 3.3 70B generates a grounded answer.

Why hybrid + rerank:
  - Pure dense retrieval misses exact keyword matches (policy codes, specific years).
    e.g. "Budget 2025 CDC Vouchers" returns 2023/2024 budget chunks because the
    topic embedding is year-agnostic. BM25 gives "2025" a strong keyword signal.
  - Cross-encoder reranking rescores all fused candidates together, catching multi-
    chunk relevance that cosine similarity underestimates.

Standalone smoke test:
    python app/rag.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
CORPUS_DIR = Path(__file__).parent.parent / "corpus"
COLLECTION_NAME = "asksg"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

TOP_K = 9    # chunks passed to LLM (optimal: faithfulness peaks at k=9 in eval)
FETCH = 25   # candidates fetched per retriever before RRF + reranking
RRF_K = 60   # standard RRF constant; higher = flatter score distribution

SYSTEM_PROMPT = """\
You are AskSG, an assistant that answers questions about Singapore public policy \
using only the provided source documents.

Rules:
- Answer based solely on the context provided. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, say so clearly.
- Be concise and direct.
- Do not make up statistics, dates, or policy details not present in the context.\
"""


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation + split. Consistent across indexing and querying."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


# ---------------------------------------------------------------------------
# Loader — call once per session and cache
# ---------------------------------------------------------------------------

def load_retriever() -> tuple:
    """
    Return (model, collection, bm25, all_chunks, reranker).

    - model      : SentenceTransformer for query embedding
    - collection : ChromaDB collection (dense index)
    - bm25       : BM25Okapi over all chunks (keyword index)
    - all_chunks : list of chunk dicts from chunks.jsonl (BM25 lookup table)
    - reranker   : CrossEncoder for re-scoring fused candidates
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}. "
            "Run `python etl/build_index.py` first."
        )
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    chunks_file = CORPUS_DIR / "chunks.jsonl"
    all_chunks = [
        json.loads(line)
        for line in chunks_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in all_chunks])
    reranker = CrossEncoder(RERANKER_MODEL)

    return model, collection, bm25, all_chunks, reranker


# ---------------------------------------------------------------------------
# Stage 1: Baseline — dense-only retrieval (original pipeline)
# ---------------------------------------------------------------------------

def retrieve_dense(
    query: str,
    model: SentenceTransformer,
    collection,
    k: int = TOP_K,
) -> list[dict]:
    """Dense vector retrieval only — no BM25, no reranking. Original baseline."""
    emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": t,
            "source": m["source"],
            "document": m["document"],
            "score": round(1 - d, 4),
        }
        for t, m, d in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


# ---------------------------------------------------------------------------
# Stage 1+2: Hybrid retrieval (BM25 + dense) fused with RRF
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    model: SentenceTransformer,
    collection,
    bm25: BM25Okapi,
    all_chunks: list[dict],
    k: int = TOP_K,
    fetch: int = FETCH,
) -> list[dict]:
    """
    Return top-k chunks from RRF fusion of dense and BM25 retrieval.

    Each retriever independently fetches `fetch` candidates.
    RRF score = 1/(RRF_K + dense_rank + 1) + 1/(RRF_K + bm25_rank + 1).
    Chunks absent from a retriever get a penalty rank of `fetch`.
    """
    # --- Dense (vector) retrieval ---
    emb = model.encode([query]).tolist()
    dense = collection.query(
        query_embeddings=emb,
        n_results=fetch,
        include=["documents", "metadatas", "distances"],
    )
    dense_ranks: dict[str, int] = {}
    dense_data: dict[str, dict] = {}
    for rank, (cid, text, meta, dist) in enumerate(zip(
        dense["ids"][0],
        dense["documents"][0],
        dense["metadatas"][0],
        dense["distances"][0],
    )):
        dense_ranks[cid] = rank
        dense_data[cid] = {
            "text": text,
            "source": meta["source"],
            "document": meta["document"],
            "score": round(1 - dist, 4),
        }

    # --- BM25 (keyword) retrieval ---
    tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(tokens)
    top_bm25_idx = bm25_scores.argsort()[::-1][:fetch]
    bm25_ranks: dict[str, int] = {}
    bm25_data: dict[str, dict] = {}
    for rank, idx in enumerate(top_bm25_idx):
        c = all_chunks[idx]
        bm25_ranks[c["chunk_id"]] = rank
        bm25_data[c["chunk_id"]] = {
            "text": c["text"],
            "source": c["source"],
            "document": c["document"],
        }

    # --- RRF fusion ---
    all_ids = set(dense_ranks) | set(bm25_ranks)
    rrf: dict[str, float] = {
        cid: (
            1 / (RRF_K + dense_ranks.get(cid, fetch) + 1) +
            1 / (RRF_K + bm25_ranks.get(cid, fetch) + 1)
        )
        for cid in all_ids
    }
    top_ids = sorted(all_ids, key=lambda x: rrf[x], reverse=True)[:k]

    return [
        {**(dense_data.get(cid) or bm25_data[cid]), "score": round(rrf[cid], 6)}
        for cid in top_ids
    ]


# ---------------------------------------------------------------------------
# Stage 3: Cross-encoder reranking
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    chunks: list[dict],
    reranker: CrossEncoder,
    top_n: int = TOP_K,
) -> list[dict]:
    """
    Re-score chunks with a cross-encoder (query + chunk jointly encoded).
    Returns top_n chunks sorted by cross-encoder relevance score.
    """
    if not chunks:
        return chunks
    scores = reranker.predict([(query, c["text"]) for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    # Replace score with cross-encoder score so callers see the final ranking signal
    return [{**c, "score": round(float(s), 4)} for s, c in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Stage 4: Generate
# ---------------------------------------------------------------------------

def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        label = f"[{i}] {chunk['source'].upper()} / {chunk['document']}"
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def answer(
    query: str,
    model: SentenceTransformer,
    collection,
    bm25: BM25Okapi,
    all_chunks: list[dict],
    reranker: CrossEncoder,
    k: int = TOP_K,
    mode: str = "hybrid",
) -> dict:
    """
    Full RAG pipeline. Returns answer + sources.

    mode="hybrid"   — BM25 + dense + RRF → cross-encoder rerank (production)
    mode="baseline" — dense-only, no rerank (original pipeline, used for eval comparison)
    """
    if mode == "hybrid":
        chunks = retrieve(query, model, collection, bm25, all_chunks, k=FETCH)
        chunks = rerank(query, chunks, reranker, top_n=k)
    else:
        chunks = retrieve_dense(query, model, collection, k=k)
    context = _build_context(chunks)

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
) -> tuple:
    """Streaming pipeline. Returns (groq_stream, chunks) for Streamlit write_stream."""
    chunks = retrieve(query, model, collection, bm25, all_chunks, k=FETCH)
    chunks = rerank(query, chunks, reranker, top_n=TOP_K)
    context = _build_context(chunks)

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


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading retriever (dense + BM25 + reranker)...")
    _model, _collection, _bm25, _all_chunks, _reranker = load_retriever()
    print(f"Collection: {_collection.count():,} chunks  |  BM25: {len(_all_chunks):,} chunks\n")

    # Q5 temporal disambiguation — was the hardest case in baseline eval
    _query = "What CDC Voucher amount was announced in Budget 2025 and when will it be disbursed?"
    print(f"Q: {_query}")
    _result = answer(_query, _model, _collection, _bm25, _all_chunks, _reranker)
    print(f"\nA: {_result['answer']}")
    print(f"\nSources (reranked):")
    for c in _result["sources"]:
        print(f"  [{c['score']:.6f}] {c['source']}/{c['document']}")
