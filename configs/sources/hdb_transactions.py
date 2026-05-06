"""
Source definition: HDB Resale Flat Prices dataset from data.gov.sg.

This is structured tabular data fetched via the CKAN datastore API.
It feeds the Agentic RAG tool (hdb_resale_query), not the vector index.

Dataset: https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view
"""

DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
API_URL = "https://data.gov.sg/api/action/datastore_search"
RECORDS_PER_PAGE = 10_000
