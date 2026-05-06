"""
Cross-encoder reranker: jointly encode (query, chunk) pairs and re-score.

Why reranking on top of hybrid retrieval:
  Cosine similarity and BM25 score each chunk independently against the query.
  A cross-encoder sees query and chunk together, catching relevance signals
  that bi-encoder similarity misses (e.g. multi-chunk co-reference).
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder


def rerank(
    query: str,
    chunks: list[dict],
    reranker: CrossEncoder,
    top_n: int,
) -> list[dict]:
    """
    Re-score `chunks` with the cross-encoder and return the top-n by relevance.
    The `score` field is replaced with the cross-encoder logit.
    """
    if not chunks:
        return chunks
    scores = reranker.predict([(query, c["text"]) for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [{**c, "score": round(float(s), 4)} for s, c in ranked[:top_n]]
