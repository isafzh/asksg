"""
Pipeline: chunk extracted text and build both retrieval indexes.

  [data/interim/extracted_text/**/*.txt]
       → [src/processing/clean_text.py + src/processing/chunker.py]
       → [data/processed/chunks.jsonl]
       → [src/indexing/build_vector_index.py]  → [data/indexes/chroma/]
       → [src/indexing/build_keyword_index.py] → (in-memory at query time)

Usage:
    python pipelines/build_indexes.py
    python pipelines/build_indexes.py --force   # force re-embed even if count matches
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.processing.clean_text import clean
from src.processing.chunker import split_text
from src.indexing.build_vector_index import build as build_vector

EXTRACTED_TEXT_DIR = ROOT / "data" / "interim" / "extracted_text"
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"


def chunk_all() -> list[dict]:
    txt_files = sorted(EXTRACTED_TEXT_DIR.rglob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {EXTRACTED_TEXT_DIR.relative_to(ROOT)}")
        print("Run: python pipelines/ingest_documents.py")
        return []

    all_chunks: list[dict] = []
    for path in txt_files:
        source = path.parent.name
        document = path.stem
        text = clean(path.read_text(encoding="utf-8"))

        if len(text) < 100:
            print(f"  SKIP (too short): {path.relative_to(EXTRACTED_TEXT_DIR)}")
            continue

        chunks = split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{source}__{document}__{i:04d}",
                "source": source,
                "document": document,
                "text": chunk,
            })

        print(f"  {source}/{document}: {len(text):,} chars -> {len(chunks)} chunks")

    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-embedding")
    args = parser.parse_args()

    # Step 1: chunk
    print("=== Chunking documents ===")
    chunks = chunk_all()
    if not chunks:
        return

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
        encoding="utf-8",
    )
    lengths = [len(c["text"]) for c in chunks]
    print(f"\nTotal: {len(chunks)} chunks from {len(set(c['document'] for c in chunks))} documents")
    print(f"Length — min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)//len(lengths)}")
    print(f"Saved -> {CHUNKS_FILE.relative_to(ROOT)}")

    # Step 2: embed + vector index
    print("\n=== Building vector index ===")
    build_vector(force=args.force)

    print("\n=== Done ===")
    print("BM25 index is built in-memory at query time from chunks.jsonl.")


if __name__ == "__main__":
    main()
