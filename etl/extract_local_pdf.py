"""
Extracts text from a manually downloaded PDF and saves to corpus/.

Usage:
    python etl/extract_local_pdf.py <pdf_path> <source> <name>

Example:
    python etl/extract_local_pdf.py data/raw_pdfs/budget2025.pdf budget budget_2025_speech
"""

import sys
import io
import pdfplumber
from pathlib import Path

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def extract(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python etl/extract_local_pdf.py <pdf_path> <source> <name>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    source = sys.argv[2]
    name = sys.argv[3]

    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"Extracting: {pdf_path}")
    text = extract(pdf_path).strip()
    print(f"Extracted {len(text):,} chars")

    out_dir = CORPUS_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
