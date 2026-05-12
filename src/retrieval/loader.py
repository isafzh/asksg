"""
Session-level loader: initialise all retrieval components once and cache them.

Returns a named tuple so callers can access components by name rather than
by positional index.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexing.build_keyword_index import build as build_bm25, load_chunks

CHROMA_DIR = Path(os.getenv("ASKSG_CHROMA_DIR", ROOT / "data" / "indexes" / "chroma"))
COLLECTION_NAME = "asksg"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Retriever(NamedTuple):
    model: SentenceTransformer
    collection: chromadb.Collection
    bm25: BM25Okapi
    chunks: list[dict]
    reranker: CrossEncoder


def load() -> Retriever:
    """
    Load all retrieval components.  Call once per session; the result is
    safe to cache (e.g. via @st.cache_resource in Streamlit).
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Vector index not found at {CHROMA_DIR}. "
            "Run: python pipelines/build_indexes.py"
        )

    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    all_chunks = load_chunks()
    bm25, _ = build_bm25(chunks=all_chunks)
    reranker = CrossEncoder(RERANKER_MODEL)

    return Retriever(
        model=model,
        collection=collection,
        bm25=bm25,
        chunks=all_chunks,
        reranker=reranker,
    )
