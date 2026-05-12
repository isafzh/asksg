"""
Vector index builder: embed all chunks and store in ChromaDB.

Input:  data/processed/chunks.jsonl
Output: data/indexes/chroma/  (persisted ChromaDB collection)

Re-running is safe — counts are compared and re-indexing only triggers
when the chunk count has changed.  Delete data/indexes/chroma/ to force
a full rebuild.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = ROOT / "data" / "indexes" / "chroma"
COLLECTION_NAME = "asksg"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def load_chunks() -> list[dict]:
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(force: bool = False) -> None:
    if not CHUNKS_FILE.exists():
        print("chunks.jsonl not found. Run: python pipelines/build_indexes.py")
        return

    chunks = load_chunks()
    print(f"Loaded {len(chunks):,} chunks from {CHUNKS_FILE.name}")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("(First run downloads ~90 MB — subsequent runs are instant)")
    model = SentenceTransformer(EMBED_MODEL)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if not force and existing == len(chunks):
        print(f"\nCollection already has {existing} chunks — nothing to do.")
        print("Pass force=True or delete data/indexes/chroma/ to rebuild.")
        return
    elif existing > 0:
        print(f"\nChunk count changed ({existing} → {len(chunks)}) — rebuilding.")
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    print(f"\nEmbedding {len(chunks):,} chunks in batches of {BATCH_SIZE}...")
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        embeddings = model.encode([c["text"] for c in batch], show_progress_bar=False).tolist()
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "document": c["document"]} for c in batch],
        )
        print(f"  {min(start + BATCH_SIZE, len(chunks)):,} / {len(chunks):,}", end="\r")

    print()
    print(f"Done. {collection.count():,} chunks stored in {CHROMA_DIR.relative_to(ROOT)}")

    print("\nSanity check — querying: 'CPF housing grant eligibility'")
    test_emb = model.encode(["CPF housing grant eligibility"]).tolist()
    results = collection.query(query_embeddings=test_emb, n_results=3)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n  Result {i+1} [{meta['source']}/{meta['document']}]:")
        print(f"  {doc[:200]}...")


if __name__ == "__main__":
    build()
