"""
Embeds all chunks from corpus/chunks.jsonl using sentence-transformers
and stores them in a ChromaDB collection.

Model: all-MiniLM-L6-v2 (local, free, 256-token limit)
DB:    chroma_db/  (persisted to disk, git-ignored)

Usage:
    python etl/build_index.py

Re-running is safe — it checks if the collection already has the same
number of chunks and skips re-embedding if nothing changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
CHUNKS_FILE = CORPUS_DIR / "chunks.jsonl"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

COLLECTION_NAME = "asksg"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64  # embed this many chunks at a time


def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    if not CHUNKS_FILE.exists():
        print("chunks.jsonl not found. Run etl/chunk_documents.py first.")
        return

    chunks = load_chunks()
    print(f"Loaded {len(chunks):,} chunks from {CHUNKS_FILE.name}")

    # --- Load embedding model ---
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("(First run downloads ~90MB — subsequent runs are instant)")
    model = SentenceTransformer(EMBED_MODEL)

    # --- Connect to ChromaDB ---
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Skip if already indexed
    existing = collection.count()
    if existing == len(chunks):
        print(f"\nCollection already has {existing} chunks. Nothing to do.")
        print("Delete chroma_db/ to force re-index.")
        return
    elif existing > 0:
        print(f"\nCollection has {existing} chunks but {len(chunks)} in file — re-indexing.")
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # --- Embed and store in batches ---
    print(f"\nEmbedding {len(chunks):,} chunks in batches of {BATCH_SIZE}...")

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"source": c["source"], "document": c["document"]}
                for c in batch
            ],
        )

        done = min(start + BATCH_SIZE, len(chunks))
        print(f"  {done:,} / {len(chunks):,}", end="\r")

    print(f"\nDone. {collection.count():,} chunks stored in {CHROMA_DIR.name}/")

    # --- Quick sanity check ---
    print("\nSanity check — querying: 'CPF housing grant eligibility'")
    test_embedding = model.encode(["CPF housing grant eligibility"]).tolist()
    results = collection.query(query_embeddings=test_embedding, n_results=3)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n  Result {i+1} [{meta['source']}/{meta['document']}]:")
        print(f"  {doc[:200]}...")


if __name__ == "__main__":
    main()
