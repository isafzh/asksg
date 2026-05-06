"""
Source registry for structured data (HDB resale transaction records).

HDB Resale Flat Prices (2017–present) from data.gov.sg — a tabular dataset of
180 000+ transactions with columns: month, town, flat_type, storey_range,
floor_area_sqm, resale_price, remaining_lease, etc.

This dataset is fetched via the data.gov.sg CKAN datastore API (paginated),
cleaned into a typed parquet file, and queried by the Agentic RAG tool
(src/tools/hdb_resale_query.py) to answer numeric/statistical questions about
resale prices — a separate retrieval modality from the policy document pipeline.

Maps to: data/raw/structured/              (raw API export)
         data/interim/cleaned_tables/      (typed CSV after cleaning)
         data/processed/hdb_resale.parquet (final query-ready table)

Dataset: https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view
"""

DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
API_URL = "https://data.gov.sg/api/action/datastore_search"
RECORDS_PER_PAGE = 10_000
