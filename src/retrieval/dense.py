"""Dense vector retrieval: embed query and find nearest neighbours in ChromaDB."""

from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer


def retrieve_dense(
    query: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    k: int,
) -> list[dict]:
    """Return top-k chunks by cosine similarity. Score = 1 - cosine_distance."""
    emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "chunk_id": cid,
            "text": text,
            "source": meta["source"],
            "document": meta["document"],
            "score": round(1 - dist, 4),
        }
        for cid, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
