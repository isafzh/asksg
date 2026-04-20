from __future__ import annotations

"""
Downloads Singapore government documents for the RAG corpus.

Output: corpus/<source>/<name>.txt  (tracked by git, safe for deployment)

Sources:
- Singapore Budget speeches (singaporebudget.gov.sg)
- MAS monetary policy statements (mas.gov.sg) — JavaScript-rendered, needs Playwright
- HDB eligibility guides (hdb.gov.sg)
- CPF contribution guides (cpf.gov.sg)

Usage:
    python etl/fetch_documents.py

For JavaScript-rendered sites, install Playwright first:
    pip install playwright && playwright install chromium
"""

import io
import time
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup

CORPUS_DIR = Path(__file__).parent.parent / "corpus"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Document registry
# Each entry: dict with keys: source, name, urls (list, tried in order), type
# type: "pdf" | "html" | "html_js" (JavaScript-rendered, needs Playwright)
# ---------------------------------------------------------------------------
DOCUMENTS = [
    # --- Budget speeches (PDFs via mof.gov.sg) ---
    {
        "source": "budget",
        "name": "budget_2025_speech",
        "type": "pdf",
        "urls": [
            # CMS asset confirmed working (from singaporebudget.gov.sg/budget-speech/budget-statement)
            "https://cms.singaporebudget.gov.sg/assets/a2f09ed4-d17b-401f-b4eb-4f89e2b00a86",
            "https://www.mof.gov.sg/docs/librariesprovider3/budget2025/download/pdf/fy2025_budget_statement.pdf",
        ],
    },
    {
        "source": "budget",
        "name": "budget_2024_speech",
        "type": "pdf",
        "urls": [
            "https://isomer-user-content.by.gov.sg/153/ba1b1554-123d-4e2b-b98f-cc3551c3d6e3/fy2024_budget_statement.pdf",
        ],
    },
    {
        "source": "budget",
        "name": "budget_2023_speech",
        "type": "pdf",
        "urls": [
            "https://isomer-user-content.by.gov.sg/153/5b06afae-697f-4183-8d9c-7ed511dda591/fy2023_budget_statement.pdf",
        ],
    },

    # --- MAS Macroeconomic Reviews (PDFs — each includes the Monetary Policy Statement) ---
    {
        "source": "mas",
        "name": "mas_macro_review_apr2025",
        "type": "pdf",
        "urls": [
            "https://www.mas.gov.sg/-/media/mas-media-library/publications/macroeconomic-review/2025/apr/mrapr25.pdf",
        ],
    },
    {
        "source": "mas",
        "name": "mas_macro_review_oct2024",
        "type": "pdf",
        "urls": [
            "https://www.mas.gov.sg/-/media/mas-media-library/publications/macroeconomic-review/2024/oct/mroct24.pdf",
        ],
    },
    {
        "source": "mas",
        "name": "mas_macro_review_apr2024",
        "type": "pdf",
        "urls": [
            "https://www.mas.gov.sg/-/media/mas-media-library/publications/macroeconomic-review/2024/apr/mrapr24.pdf",
        ],
    },

    # --- HDB eligibility and policies (HTML — accessible without Playwright) ---
    {
        "source": "hdb",
        "name": "hdb_buying_flat",
        "type": "html",
        "urls": [
            "https://www.hdb.gov.sg/buying-a-flat",
        ],
    },
    {
        "source": "hdb",
        "name": "hdb_resale_eligibility",
        "type": "html",
        "urls": [
            "https://www.hdb.gov.sg/buying-a-flat/resale-flats/process-for-buying-a-resale-flat/overview",
            "https://www.hdb.gov.sg/business-partners/estate-agents-and-salespersons/guide-for-estate-agents-buying-a-flat/eligibility",
        ],
    },
    {
        "source": "hdb",
        "name": "hdb_eligibility_couples_families",
        "type": "html",
        "urls": [
            "https://www.hdb.gov.sg/buying-a-flat/flat-grant-and-loan-eligibility/couples-and-families",
        ],
    },

    # --- CPF guides (HTML) ---
    {
        "source": "cpf",
        "name": "cpf_contribution_rates",
        "type": "html",
        "urls": [
            "https://www.cpf.gov.sg/employer/employer-obligations/how-much-cpf-contributions-to-pay",
        ],
    },
    {
        "source": "cpf",
        "name": "cpf_housing_usage",
        "type": "html",
        "urls": [
            "https://www.cpf.gov.sg/member/home-ownership/using-your-cpf-to-buy-a-home",
        ],
    },
]


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_pdf(content: bytes) -> str:
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)


def extract_html(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    return (main or soup).get_text(separator="\n", strip=True)


def extract_html_js(url: str) -> str | None:
    """Render JavaScript-heavy pages using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  Playwright not installed. Run: pip install playwright && playwright install chromium")
        return None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=30000, wait_until="networkidle")
            page.wait_for_timeout(2000)
            content = page.content()
        finally:
            browser.close()
    return extract_html(content.encode())


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_pdf(urls: list[str]) -> bytes | None:
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                print(f"  OK: {url}")
                return r.content
            print(f"  HTTP {r.status_code}: {url}")
        except requests.RequestException as e:
            print(f"  Error: {e}")
    return None


def save(source: str, name: str, text: str) -> Path:
    out_dir = CORPUS_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    success, failed, skipped = [], [], []

    for doc in DOCUMENTS:
        name = doc["name"]
        source = doc["source"]
        doc_type = doc["type"]
        urls = doc["urls"]

        # Skip if already fetched
        out_path = CORPUS_DIR / source / f"{name}.txt"
        if out_path.exists() and out_path.stat().st_size > 500:
            print(f"Already exists, skipping: {name}")
            skipped.append(name)
            continue

        print(f"\nFetching [{doc_type}]: {name}")

        text = None

        if doc_type == "pdf":
            content = fetch_pdf(urls)
            if content:
                text = extract_pdf(content)

        elif doc_type == "html":
            content = fetch_pdf(urls)  # reuse simple GET
            if content:
                text = extract_html(content)

        elif doc_type == "html_js":
            text = extract_html_js(urls[0])

        if not text or len(text.strip()) < 200:
            print(f"  FAILED or too short — needs manual download")
            failed.append((name, urls[0]))
            time.sleep(1)
            continue

        path = save(source, name, text.strip())
        print(f"  Saved {len(text):,} chars → {path.relative_to(Path(__file__).parent.parent)}")
        success.append(name)
        time.sleep(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"Done: {len(success)} succeeded, {len(failed)} failed, {len(skipped)} skipped")

    if failed:
        print("\nManual download needed for:")
        for name, url in failed:
            doc = next(d for d in DOCUMENTS if d["name"] == name)
            print(f"  [{doc['source']}] {name}")
            print(f"    Visit: {url}")
            if doc["type"] == "pdf":
                print(f"    → Download PDF → run: python etl/extract_local_pdf.py data/raw_pdfs/<file>.pdf {doc['source']} {name}")
            else:
                print(f"    → Copy page text → save as: corpus/{doc['source']}/{name}.txt")


if __name__ == "__main__":
    main()
