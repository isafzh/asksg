"""
Metadata-aware query filter: detect source/year signals and return a
ChromaDB where-clause to constrain dense retrieval.

Conservative design principles:
- Only filter when ONE source is unambiguously implied.
- Cross-domain queries (multiple sources detected) get no filter.
- "budget" without a year is treated as a cross-domain inhibitor.
- Ambiguous or broad queries get no filter (return None).
"""

from __future__ import annotations

import re

# Budget year: "Budget 2023/2024/2025/2026" (case-insensitive)
_BUDGET_YEAR_RE = re.compile(r"\bbudget\s+(202[3-6])\b", re.IGNORECASE)

# Keyword signals per source (all lowercase; padded query used for matching)
_SOURCE_SIGNALS: dict[str, list[str]] = {
    "cpf": [
        " cpf ", "central provident fund", "cpf life", "medisave",
        "ordinary account", "special account", "retirement sum",
        "cpf contribution", "cpf savings",
    ],
    "hdb": [
        " hdb ", "housing board", "hdb flat", "resale flat",
        " bto ", "hdb eligibility",
    ],
    "mas": [
        "monetary authority", "mas monetary", "mas's",
    ],
    "srs": [
        " srs ", "supplementary retirement scheme",
    ],
    "ssb": [
        " ssb ", "singapore savings bond",
    ],
}


def detect_filter(query: str) -> dict | None:
    """
    Return a ChromaDB where-clause filter if the query unambiguously targets
    one source or one specific Budget-year document.

    Returns None when:
    - No source signal is found
    - Multiple sources are implied (cross-domain query)
    - Query mentions "budget" (any year) alongside another source
    """
    q_padded = " " + query.lower() + " "

    budget_year_match = _BUDGET_YEAR_RE.search(query)
    has_budget_word   = "budget" in q_padded
    non_budget_matched = [
        src for src, signals in _SOURCE_SIGNALS.items()
        if any(sig in q_padded for sig in signals)
    ]

    # Cross-domain: budget signal co-occurs with another source → no filter
    if (budget_year_match or has_budget_word) and non_budget_matched:
        return None

    # Multiple non-budget sources → no filter
    if len(non_budget_matched) > 1:
        return None

    # Single specific Budget year, no other source signals
    if budget_year_match and not non_budget_matched:
        year = budget_year_match.group(1)
        return {
            "$and": [
                {"source": {"$eq": "budget"}},
                {"document": {"$eq": f"budget_{year}_speech"}},
            ]
        }

    # Single non-budget source, no budget signal
    if len(non_budget_matched) == 1 and not budget_year_match and not has_budget_word:
        return {"source": {"$eq": non_budget_matched[0]}}

    return None
