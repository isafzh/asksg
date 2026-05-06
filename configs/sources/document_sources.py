"""
Source registry for policy documents (unstructured text corpus).

Covers Singapore government publications: Budget speeches (MoF), CPF guides,
HDB eligibility pages, MAS Macroeconomic Reviews, SRS overview (IRAS), and
Singapore Savings Bonds (MAS).  These documents are fetched, extracted to
plain text, chunked, and indexed into the vector store for policy Q&A retrieval.

Each entry declares what to fetch — no fetch or parse logic lives here.
The ingestion pipeline (src/ingestion/documents/pipeline.py) reads this list
and routes each document through the appropriate fetcher and extractor.

Maps to: data/raw/documents/             (downloaded raw files)
         data/interim/extracted_text/    (parsed .txt output)

type: "pdf"     — downloaded as bytes via http_fetcher, parsed by pdf_extractor
      "html"    — downloaded as bytes via http_fetcher, parsed by html_extractor
      "html_js" — JavaScript-rendered via playwright_fetcher, parsed by html_extractor
"""

DOCUMENTS: list[dict] = [
    # --- Budget speeches (PDFs via mof.gov.sg) ---
    {
        "source": "budget",
        "name": "budget_2026_speech",
        "type": "pdf",
        "urls": [
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
    # HDB redesigned their site; current paths are under /flat-grant-and-loan-eligibility/.
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

    # --- CPF voluntary top-ups (Retirement Topping-Up Scheme) ---
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
