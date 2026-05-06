"""
Fallback pipeline: extract text from a manually downloaded PDF.

Use this when fetch_documents fails for a PDF (e.g. behind a login or CAPTCHA).
Download the PDF manually, then run this script to extract and save it.

Usage:
    python pipelines/extract_local_pdf.py <pdf_path> <source> <name>

Example:
    python pipelines/extract_local_pdf.py data/raw/documents/budget2025.pdf budget budget_2025_speech
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.documents.extractors.pdf_extractor import extract_pdf

OUTPUT_DIR = ROOT / "data" / "interim" / "extracted_text"


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    source = sys.argv[2]
    name = sys.argv[3]

    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"Extracting: {pdf_path}")
    text = extract_pdf(pdf_path.read_bytes()).strip()
    print(f"Extracted {len(text):,} chars")

    out_dir = OUTPUT_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
