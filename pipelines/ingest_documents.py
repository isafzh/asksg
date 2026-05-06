"""
Pipeline: fetch and extract all policy documents.

  [configs/sources/document_sources.py]
       → [fetchers: http / playwright]
       → [extractors: pdf / html]
       → [data/interim/extracted_text/<source>/<name>.txt]

Usage:
    python pipelines/ingest_documents.py
    python pipelines/ingest_documents.py --force   # re-fetch even if file exists
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.documents.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fetch existing files")
    args = parser.parse_args()
    run(skip_existing=not args.force)


if __name__ == "__main__":
    main()
