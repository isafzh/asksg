"""
Document ingestion pipeline: source registry → fetch → extract → save.

Reads DOCUMENTS from configs/sources/policy_docs.py and routes each entry
through the appropriate fetcher and extractor.  Extracted text is saved to
data/interim/extracted_text/<source>/<name>.txt.

Called by pipelines/ingest_documents.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# --- make project root importable when run directly ---
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sources.policy_docs import DOCUMENTS
from src.ingestion.documents.fetchers.http_fetcher import fetch_url
from src.ingestion.documents.fetchers.playwright_fetcher import fetch_with_playwright
from src.ingestion.documents.extractors.pdf_extractor import extract_pdf
from src.ingestion.documents.extractors.html_extractor import extract_html

EXTRACTED_TEXT_DIR = ROOT / "data" / "interim" / "extracted_text"
MIN_CHARS = 200  # discard anything shorter as a failed fetch


def _save(source: str, name: str, text: str) -> Path:
    out_dir = EXTRACTED_TEXT_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def run(skip_existing: bool = True) -> dict[str, list]:
    """
    Fetch and extract every document in DOCUMENTS.

    Returns a summary dict with keys: success, failed, skipped.
    Each failed entry is (name, first_url).
    """
    results: dict[str, list] = {"success": [], "failed": [], "skipped": []}

    for doc in DOCUMENTS:
        name = doc["name"]
        source = doc["source"]
        doc_type = doc["type"]
        urls: list[str] = doc["urls"]

        out_path = EXTRACTED_TEXT_DIR / source / f"{name}.txt"
        if skip_existing and out_path.exists() and out_path.stat().st_size > 500:
            print(f"Already exists, skipping: {name}")
            results["skipped"].append(name)
            continue

        print(f"\nFetching [{doc_type}]: {name}")
        text: str | None = None

        if doc_type == "pdf":
            raw = fetch_url(urls)
            if raw:
                text = extract_pdf(raw)

        elif doc_type == "html":
            raw = fetch_url(urls)
            if raw:
                text = extract_html(raw)

        elif doc_type == "html_js":
            for url in urls:
                raw = fetch_with_playwright(url)
                if raw:
                    text = extract_html(raw)
                    if text and len(text.strip()) >= MIN_CHARS:
                        break

        if not text or len(text.strip()) < MIN_CHARS:
            print(f"  FAILED or too short — needs manual download")
            results["failed"].append((name, urls[0]))
            time.sleep(1)
            continue

        path = _save(source, name, text.strip())
        print(f"  Saved {len(text):,} chars -> {path.relative_to(ROOT)}")
        results["success"].append(name)
        time.sleep(1)

    _print_summary(results)
    return results


def _print_summary(results: dict[str, list]) -> None:
    s, f, sk = len(results["success"]), len(results["failed"]), len(results["skipped"])
    print(f"\n{'='*60}")
    print(f"Done: {s} succeeded, {f} failed, {sk} skipped")

    if results["failed"]:
        print("\nManual download needed for:")
        for name, url in results["failed"]:
            doc = next(d for d in DOCUMENTS if d["name"] == name)
            print(f"  [{doc['source']}] {name}")
            print(f"    Visit: {url}")
            if doc["type"] == "pdf":
                print(f"    -> Download PDF -> run: python pipelines/extract_local_pdf.py data/raw/documents/<file>.pdf {doc['source']} {name}")
            else:
                print(f"    -> Copy page text -> save as: data/interim/extracted_text/{doc['source']}/{name}.txt")
