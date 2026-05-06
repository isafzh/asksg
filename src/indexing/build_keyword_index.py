"""
Keyword index builder: construct a BM25Okapi index over all chunks.

BM25 is rebuilt in-memory at startup (takes ~1 s for ~2 000 chunks).
This module exposes the build function so retrieval/loader.py can call it
without duplicating the construction logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.bm25_retriever import tokenize

CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"


def load_chunks(path: Path = CHUNKS_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(chunks: list[dict] | None = None) -> tuple[BM25Okapi, list[dict]]:
    """
    Build and return (bm25_index, all_chunks).

    If `chunks` is provided (already loaded), it is used directly.
    Otherwise chunks are loaded from CHUNKS_FILE.
    """
    if chunks is None:
        chunks = load_chunks()
    corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus), chunks
