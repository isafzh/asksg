"""BM25 keyword retrieval: tokenise query, score all chunks, return top-k."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation + split. Consistent across indexing and querying."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def retrieve_bm25(
    query: str,
    bm25: BM25Okapi,
    all_chunks: list[dict],
    k: int,
) -> list[dict]:
    """Return top-k chunks by BM25 score with rank attached."""
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_idx = scores.argsort()[::-1][:k]
    return [
        {**all_chunks[i], "bm25_score": round(float(scores[i]), 6)}
        for i in top_idx
    ]
