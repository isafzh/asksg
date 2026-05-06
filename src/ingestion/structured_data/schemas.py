"""
Column definitions for the HDB Resale Flat Prices dataset.

Used by clean_hdb_resale.py for type coercion and by hdb_resale_query.py
for building structured queries against the CSV.
"""

# Columns returned by the data.gov.sg API
RAW_COLUMNS = [
    "_id",
    "month",            # "YYYY-MM"
    "town",
    "flat_type",        # "3 ROOM", "4 ROOM", etc.
    "block",
    "street_name",
    "storey_range",     # "01 TO 03"
    "floor_area_sqm",
    "flat_model",
    "lease_commence_date",
    "remaining_lease",  # "XX years YY months"
    "resale_price",
]

# Dtypes after cleaning (used in clean_hdb_resale.py)
COLUMN_TYPES = {
    "month": "str",
    "town": "str",
    "flat_type": "str",
    "block": "str",
    "street_name": "str",
    "storey_range": "str",
    "floor_area_sqm": "float",
    "flat_model": "str",
    "lease_commence_date": "int",
    "resale_price": "float",
}
