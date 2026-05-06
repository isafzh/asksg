"""System prompt and context formatting for the generation step."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are AskSG, an assistant that answers questions about Singapore public policy \
using only the provided source documents.

Rules:
- Answer based solely on the context provided. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, say so clearly.
- Be concise and direct.
- Do not make up statistics, dates, or policy details not present in the context.\
"""


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        label = f"[{i}] {chunk['source'].upper()} / {chunk['document']}"
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
