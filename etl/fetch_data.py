"""
Fetches HDB Resale Flat Prices (2017-present) from data.gov.sg
and saves to data/hdb_resale.csv.

Dataset ID: d_8b84c4ee58e3cfc0ece0d773c8ca6abc
Docs: https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view
"""

import time
import requests
import pandas as pd
from pathlib import Path

DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
API_URL = f"https://data.gov.sg/api/action/datastore_search"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "hdb_resale.csv"
LIMIT = 10000  # records per page


def fetch_all_records() -> pd.DataFrame:
    all_records = []
    offset = 0

    print("Fetching HDB resale data from data.gov.sg...")

    while True:
        for attempt in range(5):
            response = requests.get(
                API_URL,
                params={
                    "resource_id": DATASET_ID,
                    "limit": LIMIT,
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
        result = response.json()["result"]

        records = result["records"]
        total = result["total"]

        all_records.extend(records)
        offset += len(records)

        print(f"  Fetched {offset:,} / {total:,} records")

        if offset >= total:
            break

    return pd.DataFrame(all_records)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_all_records()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} records to {OUTPUT_PATH}")
    print("\nColumns:", list(df.columns))
    print("\nSample row:")
    print(df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
