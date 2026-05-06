"""Text normalisation applied to every extracted document before chunking."""

from __future__ import annotations

import re


def clean(text: str) -> str:
    """Collapse excess whitespace and remove PDF artefacts."""
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excessive blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)   # collapse multiple spaces/tabs
    text = re.sub(r"\x00", "", text)         # strip null bytes from PDFs
    return text.strip()
