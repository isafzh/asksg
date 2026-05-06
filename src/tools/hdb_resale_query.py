"""
Agentic RAG tool: structured HDB resale transaction query.

This wraps pandas/SQL queries over the processed HDB resale parquet file
as a callable tool for an LLM agent.  The agent calls this when the
question requires numeric data — median prices, transaction counts,
price trends by town or flat type.

Status: stub — implement query logic and wire to agent framework when ready.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "processed" / "hdb_resale.parquet"

_cache: pd.DataFrame | None = None


def _load() -> pd.DataFrame:
    global _cache
    if _cache is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"HDB resale data not found at {DATA_PATH}. "
                "Run: python pipelines/ingest_hdb_data.py"
            )
        _cache = pd.read_parquet(DATA_PATH)
    return _cache


def query_hdb_resale(
    town: str | None = None,
    flat_type: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> dict:
    """
    Return summary statistics for HDB resale transactions matching the filters.

    All parameters are optional; unset filters are not applied.
    Returns: {count, median_price, min_price, max_price, sample_rows}
    """
    df = _load()

    if town:
        df = df[df["town"].str.upper() == town.upper()]
    if flat_type:
        df = df[df["flat_type"].str.upper() == flat_type.upper()]
    if year_from:
        df = df[df["year"] >= year_from]
    if year_to:
        df = df[df["year"] <= year_to]

    if df.empty:
        return {"count": 0, "message": "No transactions match the given filters."}

    return {
        "count": len(df),
        "median_price": df["resale_price"].median(),
        "min_price": df["resale_price"].min(),
        "max_price": df["resale_price"].max(),
        "sample_rows": df.head(3).to_dict(orient="records"),
    }
