"""
HDB Resale Flat Prices fetcher: paginated pull from the data.gov.sg CKAN API.

This structured dataset feeds the Agentic RAG tool (hdb_resale_query),
not the vector index.  Output: data/interim/cleaned_tables/hdb_resale.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sources.structured_sources import API_URL, DATASET_ID, RECORDS_PER_PAGE

OUTPUT_PATH = ROOT / "data" / "interim" / "cleaned_tables" / "hdb_resale.csv"


def fetch_all_records() -> pd.DataFrame:
    """Paginate through the full dataset and return a single DataFrame."""
    all_records: list[dict] = []
    offset = 0

    print("Fetching HDB resale data from data.gov.sg...")

    while True:
        for attempt in range(5):
            response = requests.get(
                API_URL,
                params={
                    "resource_id": DATASET_ID,
                    "limit": RECORDS_PER_PAGE,
                    "offset": offset,
                },
            )
            if response.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            break
        else:
            raise RuntimeError(f"Gave up after 5 retries (offset={offset})")

        result = response.json()["result"]
        records = result["records"]
        total = result["total"]

        all_records.extend(records)
        offset += len(records)
        print(f"  Fetched {offset:,} / {total:,} records")

        if offset >= total:
            break

    return pd.DataFrame(all_records)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_all_records()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} records -> {OUTPUT_PATH.relative_to(ROOT)}")
    print("Columns:", list(df.columns))
    print("\nSample row:")
    print(df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
