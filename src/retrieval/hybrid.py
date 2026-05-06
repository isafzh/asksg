"""
Hybrid retrieval: BM25 + dense fused with Reciprocal Rank Fusion (RRF).

Why RRF instead of score normalisation:
  Cosine and BM25 scores live on different scales.  RRF avoids normalising by
  using only rank positions, making the fusion scale-invariant.

  RRF score = 1/(k + dense_rank) + 1/(k + bm25_rank)
  Chunks absent from one retriever get a penalty rank of `fetch`.
"""

from __future__ import annotations

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.retrieval.bm25_retriever import tokenize

RRF_K = 60  # standard constant; higher = flatter distribution


def retrieve_hybrid(
    query: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    bm25: BM25Okapi,
    all_chunks: list[dict],
    k: int,
    fetch: int,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """
    Fuse dense and BM25 results with RRF and return top-k chunks.

    Each retriever independently fetches `fetch` candidates.
    """
    # --- Dense retrieval ---
    emb = model.encode([query]).tolist()
    dense_results = collection.query(
        query_embeddings=emb,
        n_results=fetch,
        include=["documents", "metadatas", "distances"],
    )
    dense_ranks: dict[str, int] = {}
    dense_data: dict[str, dict] = {}
    for rank, (cid, text, meta, dist) in enumerate(zip(
        dense_results["ids"][0],
        dense_results["documents"][0],
        dense_results["metadatas"][0],
        dense_results["distances"][0],
    )):
        dense_ranks[cid] = rank
        dense_data[cid] = {
            "chunk_id": cid,
            "text": text,
            "source": meta["source"],
            "document": meta["document"],
        }

    # --- BM25 retrieval ---
    tokens = tokenize(query)
    bm25_scores = bm25.get_scores(tokens)
    top_bm25_idx = bm25_scores.argsort()[::-1][:fetch]
    bm25_ranks: dict[str, int] = {}
    bm25_data: dict[str, dict] = {}
    for rank, idx in enumerate(top_bm25_idx):
        c = all_chunks[idx]
        bm25_ranks[c["chunk_id"]] = rank
        bm25_data[c["chunk_id"]] = {
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "source": c["source"],
            "document": c["document"],
        }

    # --- RRF fusion ---
    all_ids = set(dense_ranks) | set(bm25_ranks)
    rrf_scores = {
        cid: (
            1 / (rrf_k + dense_ranks.get(cid, fetch) + 1) +
            1 / (rrf_k + bm25_ranks.get(cid, fetch) + 1)
        )
        for cid in all_ids
    }
    top_ids = sorted(all_ids, key=lambda x: rrf_scores[x], reverse=True)[:k]

    return [
        {**(dense_data.get(cid) or bm25_data[cid]), "score": round(rrf_scores[cid], 6)}
        for cid in top_ids
    ]
