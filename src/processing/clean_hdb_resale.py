"""
HDB Resale CSV cleaner: type coercion and column normalisation.

Input:  data/interim/cleaned_tables/hdb_resale.csv  (raw API export)
Output: data/processed/hdb_resale.parquet           (typed, query-ready)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.structured_data.schemas import COLUMN_TYPES

INPUT_PATH = ROOT / "data" / "interim" / "cleaned_tables" / "hdb_resale.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "hdb_resale.parquet"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Drop internal API column
    df = df.drop(columns=["_id"], errors="ignore")

    # Coerce to declared types
    for col, dtype in COLUMN_TYPES.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") if dtype in ("float", "int") else df[col].astype(str).str.strip()

    # Derive year and month as integers for easy filtering
    if "month" in df.columns:
        df["year"] = df["month"].str[:4].astype(int)
        df["month_num"] = df["month"].str[5:7].astype(int)

    df = df.dropna(subset=["resale_price", "floor_area_sqm"])
    return df.reset_index(drop=True)


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"Input not found: {INPUT_PATH}")
        print("Run: python pipelines/ingest_hdb_data.py")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} rows from {INPUT_PATH.name}")

    df_clean = clean(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df_clean):,} rows -> {OUTPUT_PATH.relative_to(ROOT)}")
    print("Columns:", list(df_clean.columns))


if __name__ == "__main__":
    main()
