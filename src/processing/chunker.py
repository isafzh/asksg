"""
Recursive character-level text splitter with overlap.

Tries to split on natural boundaries (paragraph → sentence → word → character)
so chunks don't cut mid-sentence.  No external dependencies.

all-MiniLM-L6-v2 has a 256-token limit (~1 000 chars).
Default 500-char chunks with 50-char overlap fits safely within that limit.
"""

from __future__ import annotations

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split `text` into overlapping chunks respecting natural boundaries."""

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
                if len(part) > chunk_size and remaining:
                    chunks.extend(_split(part, remaining))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, SEPARATORS)

    # Apply overlap: each chunk starts `overlap` chars before the previous ended
    result: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0 or not result:
            result.append(chunk)
        else:
            prev_tail = result[-1][-overlap:] if len(result[-1]) >= overlap else result[-1]
            result.append((prev_tail + " " + chunk).strip())

    # Hard-split any chunk still over 1.5× limit
    final: list[str] = []
    for chunk in result:
        if len(chunk) <= chunk_size * 1.5:
            final.append(chunk)
        else:
            for start in range(0, len(chunk), chunk_size - overlap):
                final.append(chunk[start : start + chunk_size])

    return [c.strip() for c in final if c.strip()]
