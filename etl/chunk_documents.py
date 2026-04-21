"""
Reads all .txt files from corpus/, splits into overlapping chunks,
and saves to corpus/chunks.jsonl (one JSON object per line).

Each chunk:
    {
        "chunk_id": "budget__budget_2025_speech__0042",
        "source":   "budget",
        "document": "budget_2025_speech",
        "text":     "..."
    }

Usage:
    python etl/chunk_documents.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
OUTPUT_FILE = CORPUS_DIR / "chunks.jsonl"

# all-MiniLM-L6-v2 has a 256-token limit (~1000 chars).
# 500-char chunks with 50-char overlap is safe and gives good retrieval granularity.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse excessive blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)        # collapse multiple spaces/tabs
    text = re.sub(r"\x00", "", text)              # remove null bytes from PDFs
    return text.strip()


# ---------------------------------------------------------------------------
# Chunker — recursive character splitter (no dependencies)
# Tries to split on paragraph, then sentence, then word boundaries.
# ---------------------------------------------------------------------------

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursively split text into chunks respecting natural boundaries."""

    def _split(text: str, separators: list[str]) -> list[str]:
        sep = separators[0]
        remaining = separators[1:]

        parts = text.split(sep) if sep else list(text)
        chunks: list[str] = []
        current = ""

        for part in parts:
            piece = (current + sep + part).lstrip() if current else part
            if len(piece) <= chunk_size:
                current = piece
            else:
                if current:
                    chunks.append(current)
                # Part itself too long — recurse with next separator
                if len(part) > chunk_size and remaining:
                    chunks.extend(_split(part, remaining))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, SEPARATORS)

    # Apply overlap: each chunk starts overlap chars before the previous ended
    result: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0 or not result:
            result.append(chunk)
        else:
            prev_end = result[-1][-overlap:] if len(result[-1]) >= overlap else result[-1]
            result.append((prev_end + " " + chunk).strip())

    # Final pass: if any chunk is still over limit, hard-split it
    final: list[str] = []
    for chunk in result:
        if len(chunk) <= chunk_size * 1.5:
            final.append(chunk)
        else:
            for start in range(0, len(chunk), chunk_size - overlap):
                final.append(chunk[start : start + chunk_size])

    return [c.strip() for c in final if c.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    txt_files = sorted(CORPUS_DIR.rglob("*.txt"))

    if not txt_files:
        print("No .txt files found in corpus/. Run etl/fetch_documents.py first.")
        return

    all_chunks: list[dict] = []

    for path in txt_files:
        source = path.parent.name
        document = path.stem
        raw = path.read_text(encoding="utf-8")
        text = clean(raw)

        if len(text) < 100:
            print(f"  SKIP (too short): {path.relative_to(CORPUS_DIR)}")
            continue

        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{source}__{document}__{i:04d}",
                "source": source,
                "document": document,
                "text": chunk,
            })

        print(f"  {source}/{document}: {len(text):,} chars → {len(chunks)} chunks")

    OUTPUT_FILE.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in all_chunks),
        encoding="utf-8",
    )

    print(f"\nTotal: {len(all_chunks)} chunks from {len(txt_files)} documents")
    print(f"Saved → {OUTPUT_FILE.relative_to(Path(__file__).parent.parent)}")

    # Quick stats
    lengths = [len(c["text"]) for c in all_chunks]
    print(f"Chunk length — min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)//len(lengths)}")


if __name__ == "__main__":
    main()
