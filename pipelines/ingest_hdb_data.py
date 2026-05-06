"""
Pipeline: fetch HDB resale transaction data and clean it for the Agentic RAG tool.

  [configs/sources/hdb_transactions.py]
       → [src/ingestion/structured_data/hdb_api_fetcher.py]
       → [data/interim/cleaned_tables/hdb_resale.csv]
       → [src/processing/clean_hdb_resale.py]
       → [data/processed/hdb_resale.parquet]

Usage:
    python pipelines/ingest_hdb_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.structured_data.hdb_api_fetcher import fetch_all_records, OUTPUT_PATH
from src.processing.clean_hdb_resale import clean, INPUT_PATH, OUTPUT_PATH as CLEAN_OUTPUT


def main() -> None:
    import pandas as pd

    # Step 1: fetch
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_raw = fetch_all_records()
    df_raw.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved raw: {OUTPUT_PATH.relative_to(ROOT)}")

    # Step 2: clean
    df_clean = clean(df_raw)
    CLEAN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(CLEAN_OUTPUT, index=False)
    print(f"Saved clean: {CLEAN_OUTPUT.relative_to(ROOT)} ({len(df_clean):,} rows)")


if __name__ == "__main__":
    main()
