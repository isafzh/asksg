"""
RAG pipeline: embed query → retrieve from ChromaDB → generate with Groq.

Standalone smoke test:
    python app/rag.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "asksg"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

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
# Loader — call once and cache in the Streamlit app
# ---------------------------------------------------------------------------

def load_retriever() -> tuple:
    """Load embedding model and ChromaDB collection."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}. "
            "Run `python etl/build_index.py` first."
        )
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    return model, collection


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def retrieve(query: str, model, collection, k: int = TOP_K) -> list[dict]:
    """Return top-k chunks most relevant to query, with cosine similarity scores."""
    embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "document": meta["document"],
            "score": round(1 - dist, 4),
        })
    return chunks


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        label = f"[{i}] {chunk['source'].upper()} / {chunk['document']}"
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def answer(query: str, model, collection) -> dict:
    """Full RAG pipeline. Returns answer string and list of source chunks."""
    chunks = retrieve(query, model, collection)
    context = _build_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }


def stream_answer(query: str, model, collection) -> tuple:
    """RAG pipeline with streaming. Returns (groq_stream, chunks)."""
    chunks = retrieve(query, model, collection)
    context = _build_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
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
    print("Loading retriever...")
    _model, _collection = load_retriever()
    print(f"Collection: {_collection.count():,} chunks\n")

    _query = "What did Budget 2025 say about support for first-time homebuyers?"
    print(f"Q: {_query}")
    _result = answer(_query, _model, _collection)
    print(f"\nA: {_result['answer']}")
    print(f"\nSources retrieved:")
    for c in _result["sources"]:
        print(f"  [{c['score']:.3f}] {c['source']}/{c['document']}")
