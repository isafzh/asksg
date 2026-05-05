from __future__ import annotations

"""
Downloads Singapore government documents for the RAG corpus.

Output: corpus/<source>/<name>.txt  (tracked by git, safe for deployment)

Sources:
- Singapore Budget speeches (singaporebudget.gov.sg) — PDF
- MAS Macroeconomic Reviews (mas.gov.sg) — PDF
- HDB eligibility guides (hdb.gov.sg) — JavaScript-rendered, needs Playwright
- CPF guides (cpf.gov.sg) — HTML and JavaScript-rendered
- SRS overview (iras.gov.sg) — HTML
- SSB overview (mas.gov.sg) — HTML

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
        "name": "budget_2026_speech",
        "type": "pdf",
        "urls": [
            # CMS asset — confirmed as Budget 2026 speech ("Securing Our Future Together in a Changed World")
            "https://cms.singaporebudget.gov.sg/assets/a2f09ed4-d17b-401f-b4eb-4f89e2b00a86",
            "https://www.mof.gov.sg/docs/librariesprovider3/budget2026/download/pdf/fy2026_budget_statement.pdf",
        ],
    },
    {
        "source": "budget",
        "name": "budget_2025_speech",
        "type": "pdf",
        "urls": [
            "https://isomer-user-content.by.gov.sg/153/f1a99a9f-70ae-467e-8d2c-bbb9e02473a9/Budget%202025%20Statement.pdf",
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
    # --- HDB eligibility and grants (JavaScript-rendered — requires Playwright) ---
    # URLs confirmed from live HDB homepage (May 2026). HDB redesigned their site;
    # old /understanding-your-eligibility-and-housing-loan-options/ paths now redirect
    # to a "We have moved" page. Current paths are under /flat-grant-and-loan-eligibility/.
    {
        "source": "hdb",
        "name": "hdb_eligibility_couples_families",
        "type": "html_js",
        "urls": [
            "https://www.hdb.gov.sg/buying-a-flat/flat-grant-and-loan-eligibility/couples-and-families",
        ],
    },
    {
        "source": "hdb",
        "name": "hdb_eligibility_singles",
        "type": "html_js",
        "urls": [
            "https://www.hdb.gov.sg/buying-a-flat/flat-grant-and-loan-eligibility/singles",
        ],
    },

    # --- CPF guides ---
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
    {
        "source": "cpf",
        "name": "cpf_salary_ceiling_budget2023",
        "type": "html",
        "urls": [
            "https://www.cpf.gov.sg/member/infohub/news/media-news/budget-2023-cpf-monthly-salary-ceiling-to-be-raised-to-8000-by-2026",
        ],
    },
    {
        "source": "cpf",
        "name": "cpf_budget_highlights_2023",
        "type": "html",
        "urls": [
            "https://www.cpf.gov.sg/member/infohub/news/cpf-related-announcements/budget-highlights-2023",
        ],
    },
    # --- CPF retirement ---
    {
        "source": "cpf",
        "name": "cpf_retirement_sums",
        "type": "html_js",
        "urls": [
            # Infohub article has actual BRS/FRS/ERS dollar amounts (BRS $106,500 / FRS $213,000 / ERS $426,000 for 2025)
            "https://www.cpf.gov.sg/member/infohub/educational-resources/what-is-the-cpf-retirement-sum",
            "https://www.cpf.gov.sg/service/article/what-are-the-retirement-sums-basic-retirement-sum-brs-full-retirement-sum-frs-and-enhanced-retirement-sum-ers",
        ],
    },
    {
        "source": "cpf",
        "name": "cpf_retirement_at55",
        "type": "html",
        "urls": [
            "https://www.cpf.gov.sg/member/retirement-income/milestones/reaching-age-55",
        ],
    },
    {
        "source": "cpf",
        "name": "cpf_retirement_at65",
        "type": "html",
        "urls": [
            "https://www.cpf.gov.sg/member/retirement-income/milestones/reaching-age-65",
        ],
    },
    {
        "source": "cpf",
        "name": "cpf_life",
        "type": "html",
        "urls": [
            "https://www.cpf.gov.sg/member/retirement-income/monthly-payouts/cpf-life",
        ],
    },
    {
        "source": "cpf",
        "name": "cpf_retirement_withdrawals",
        "type": "html",
        "urls": [
            "https://www.cpf.gov.sg/member/retirement-income/retirement-withdrawals/withdrawing-for-immediate-retirement-needs",
        ],
    },
    # --- CPF Medisave ---
    {
        "source": "cpf",
        "name": "cpf_medisave_usage",
        "type": "html_js",
        "urls": [
            "https://www.cpf.gov.sg/member/healthcare-financing/using-your-medisave-savings",
        ],
    },
    {
        "source": "cpf",
        "name": "cpf_medisave_bhs",
        "type": "html_js",
        "urls": [
            "https://www.cpf.gov.sg/service/article/is-there-a-maximum-amount-that-i-can-save-in-my-medisave-account",
        ],
    },
    # --- CPF voluntary top-ups (Retirement Topping-Up Scheme — SA/RA top-up for tax relief) ---
    {
        "source": "cpf",
        "name": "cpf_topup_retirement",
        "type": "html_js",
        "urls": [
            "https://www.cpf.gov.sg/member/growing-your-savings/saving-more-with-cpf/top-up-to-enjoy-higher-retirement-payouts",
        ],
    },
    {
        "source": "cpf",
        "name": "cpf_topup_tax_relief",
        "type": "html_js",
        "urls": [
            "https://www.cpf.gov.sg/service/article/how-much-tax-relief-can-i-enjoy-when-i-make-cash-top-ups",
        ],
    },
    # --- CPF interest rates ---
    {
        "source": "cpf",
        "name": "cpf_interest_rates",
        "type": "html_js",
        "urls": [
            "https://www.cpf.gov.sg/service/article/what-are-the-cpf-interest-rates",
            "https://www.cpf.gov.sg/member/infohub/news/news-releases/cpf-interest-rates-from-1-january-to-31-march-2026-and-basic-healthcare-sum-for-2026",
        ],
    },
    # --- CPF Investment Scheme (CPFIS) ---
    {
        "source": "cpf",
        "name": "cpf_investment_scheme_options",
        "type": "html",
        "urls": [
            "https://www.cpf.gov.sg/member/growing-your-savings/earning-higher-returns/investing-your-cpf-savings/cpf-investment-scheme-options",
        ],
    },
    # --- SRS — Supplementary Retirement Scheme ---
    {
        "source": "srs",
        "name": "iras_srs_overview",
        "type": "html",
        "urls": [
            "https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/supplementary-retirement-scheme-(srs)-relief",
        ],
    },
    # --- Singapore Savings Bonds ---
    {
        "source": "ssb",
        "name": "mas_ssb_overview",
        "type": "html",
        "urls": [
            "https://www.mas.gov.sg/bonds-and-bills/singapore-savings-bonds",
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
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        try:
            page.goto(url, timeout=60000, wait_until="load")
            page.wait_for_timeout(3000)
            content = page.content()
        finally:
            browser.close()
    return extract_html(content.encode())


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_url(urls: list[str]) -> bytes | None:
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
            content = fetch_url(urls)
            if content:
                text = extract_pdf(content)

        elif doc_type == "html":
            content = fetch_url(urls)
            if content:
                text = extract_html(content)

        elif doc_type == "html_js":
            for url in urls:
                text = extract_html_js(url)
                if text and len(text.strip()) >= 200:
                    break

        if not text or len(text.strip()) < 200:
            print(f"  FAILED or too short — needs manual download")
            failed.append((name, urls[0]))
            time.sleep(1)
            continue

        path = save(source, name, text.strip())
        print(f"  Saved {len(text):,} chars -> {path.relative_to(Path(__file__).parent.parent)}")
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
                print(f"    -> Download PDF -> run: python etl/extract_local_pdf.py data/raw_pdfs/<file>.pdf {doc['source']} {name}")
            else:
                print(f"    -> Copy page text -> save as: corpus/{doc['source']}/{name}.txt")


if __name__ == "__main__":
    main()
