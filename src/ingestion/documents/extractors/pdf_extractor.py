"""PDF extraction: convert raw PDF bytes to plain text via pdfplumber."""

from __future__ import annotations

import io

import pdfplumber


def extract_pdf(content: bytes) -> str:
    """Extract text from PDF bytes, one page at a time, joined with blank lines."""
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)
