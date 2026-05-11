# AskSG

A Retrieval-Augmented Generation (RAG) assistant for Singapore public policy documents.

Ask questions in natural language about Singapore's Budget, housing policies, CPF rules, and monetary policy — and get answers grounded in official government sources.

![AskSG demo](docs/demo.gif)

---

## What It Does

AskSG lets you ask questions like:

- *"What did Budget 2025 say about support for first-time homebuyers?"*
- *"Am I eligible to buy an HDB resale flat as a Singapore PR?"*
- *"How has Singapore's fiscal policy stance changed from 2023 to 2025?"*
- *"What are the CPF contribution rates for employees above 55?"*
- *"What is MAS's current assessment of Singapore's inflation outlook?"*

Instead of reading through long government PDFs, you get a cited, grounded answer in seconds.

---

## Architecture

```
Singapore government documents
   (Budget speeches, MAS reviews, HDB guides, CPF guides)
            │
            ▼
    [ Ingestion ]                pipelines/ingest_documents.py
    Download PDFs + HTML
            │
            ▼
    [ Preprocessing + Indexing ] pipelines/build_indexes.py
    Clean + split into
    overlapping text chunks
    (500 chars, 50 overlap)
    → ChromaDB vector store
    → BM25 keyword index
            │
            ▼
    [ Retrieval ]                src/retrieval/
    User question
    → embed query (MiniLM)
    → BM25 keyword retrieval  ─┐
    → dense vector retrieval  ─┴─ RRF fusion (top-25)
    → cross-encoder rerank (top-9)
            │
            ▼
    [ Generation ]               src/generation/answer.py
    → Groq LLM (Llama 3.3 70B)
    → grounded answer
            │
            ▼
    [ Streamlit Interface ]      app/main.py
    Chat interface
```

---

## Data Sources

AskSG draws from 25 official Singapore government documents across 6 source types, totalling ~784K characters of extracted policy text.

| Source | Documents | Format | Topics covered |
|---|---|---|---|
| [Singapore Budget](https://www.singaporebudget.gov.sg) | Budget speeches 2023, 2024, 2025, 2026 | PDF | Fiscal policy, grants, cost-of-living transfers, CPF and HDB enhancements |
| [MAS](https://www.mas.gov.sg) | Macroeconomic Reviews Oct 2024, Apr 2025 | PDF | Monetary policy statements, GDP and inflation forecasts, labour market, exchange rate policy |
| [HDB](https://www.hdb.gov.sg) | Eligibility guides — couples & families; singles | JS-rendered HTML | BTO/resale eligibility rules, CPF housing grants, income ceilings, SPR household rules |
| [CPF Board](https://www.cpf.gov.sg) | Contribution rates, housing usage, OW ceiling, Budget 2023 highlights, retirement sums (BRS/FRS/ERS), milestones at 55 and 65, CPF LIFE, retirement withdrawals, CPFIS options, Medisave usage, Medisave BHS, interest rates, retirement top-up scheme, top-up tax relief | HTML | Full CPF lifecycle — contributions, housing, retirement planning, Medisave, interest rates, investment scheme |
| [IRAS](https://www.iras.gov.sg) | SRS tax relief overview | HTML | Supplementary Retirement Scheme — contribution limits, tax relief, withdrawal rules |
| [MAS](https://www.mas.gov.sg) | Singapore Savings Bonds overview | HTML | SSB features, step-up interest structure, application and redemption rules |

All documents are official Singapore government publications, freely accessible to the public.

### Why these sources?

**Budget speeches (2023–2026):** The annual Budget is the primary vehicle for fiscal policy announcements — grants, CPF rule changes, HDB grant amounts, cost-of-living transfers, and tax adjustments. Four consecutive years gives the system temporal context so it can distinguish what was announced in which year, and answer comparison questions across budgets.

**MAS Macroeconomic Reviews:** Published twice yearly alongside the Monetary Policy Statement (MPS), each review documents MAS's assessment of Singapore's growth, inflation, labour market, and external outlook. Apr 2025 is the most recent and contains 2025 growth and inflation forecasts. Oct 2024 provides the prior half-year policy context and baseline.

> [!NOTE]
> **MAS special features trimming** — This is the only document in the corpus that was partially extracted rather than used in full. The Oct 2024 review is a 100+ page academic publication. Only the main policy chapters (MPS, economic outlook, inflation, monetary and fiscal policy, data boxes) were kept. The three academic Special Feature articles — *Proceedings of the 2024 Asian Monetary Policy Forum*, *A Perspective on Inflation Targeting*, and *Globalisation is Not Dying, It's Just Changing* — were removed. These articles add ~69K chars of off-topic research content that would compete with policy chunks in retrieval without answering any realistic user query.

**HDB eligibility guides:** HDB's website is JavaScript-rendered and required Playwright for content extraction. The couples & families page (59K chars) covers eligibility by household type, CPF housing grants (Enhanced, Proximity, Step-Up), income ceilings, and the SPR 3-year rule for SPR-only households buying resale flats. The singles page (21K chars) covers the Single Singapore Citizen scheme. *HDB redesigned their site in 2025 — old URL paths now return a "We have moved" page; ETL uses the current `/flat-grant-and-loan-eligibility/` paths.*

**CPF guides (15 pages):** CPF is the most common topic for Singapore personal finance questions. Coverage spans the full CPF lifecycle: contribution rates by age band, OA savings for housing purchases, retirement sum milestones (BRS $106,500 / FRS $213,000 / ERS $426,000 for 2025), CPF LIFE monthly payouts, what happens at 55 and 65, retirement withdrawal rules, CPFIS for investing OA/SA savings, Medisave usage and the Basic Healthcare Sum ($79,000 for 2026), current interest rates (OA 2.5% / SA-MA-RA 4%), and the Retirement Top-Up Scheme (up to $16,000 tax relief per year).

**SRS (IRAS):** The Supplementary Retirement Scheme lets members voluntarily top up savings for additional tax relief — up to $15,300/year for Singapore Citizens and PRs, $35,700 for foreigners. Covers contribution caps, eligible SRS investments, and the withdrawal rules at retirement age (63).

**Singapore Savings Bonds (MAS):** Risk-free, flexible government bonds with step-up interest rates backed by the Singapore Government. The MAS page explains the application process, the $200,000 individual limit, and no-penalty early redemption — a common question from risk-averse savers.

### Corpus composition

| Source | Files | Chars | Share | Period covered |
|---|---|---|---|---|
| Budget | 4 | 345,862 | 44.1% | FY2023 – FY2026 (4 annual speeches) |
| MAS | 2 | 249,002 | 31.8% | Oct 2024 – Apr 2025 (2 half-yearly reviews) |
| CPF | 15 | 91,592 | 11.7% | 2023 – 2026 (salary ceiling from Budget 2023; interest rates, BHS current as of 2026) |
| HDB | 2 | 81,134 | 10.3% | Current as of May 2026 |
| SRS | 1 | 9,321 | 1.2% | Current as of May 2026 |
| SSB | 1 | 7,070 | 0.9% | Current as of May 2026 |
| **Total** | **25** | **783,981** | | |

Chunks are 500-character overlapping windows with 50-char overlap, generated by `pipelines/build_indexes.py`.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.9+ |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Keyword index | `rank-bm25` — BM25Okapi over all corpus chunks |
| Vector store | ChromaDB (local, persistent) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API — `llama-3.3-70b-versatile` (free tier) |
| PDF parsing | `pdfplumber` |
| Frontend | Streamlit |
| Deployment *(planned)* | FastAPI + Docker + AWS EC2 |

---

## Project Structure

```
asksg/
│
├── configs/                             # What to fetch and how to configure retrieval
│   ├── sources/
│   │   ├── document_sources.py          # Policy documents registry (Budget, CPF, HDB, MAS, SRS, SSB)
│   │   └── structured_sources.py        # HDB resale transaction dataset (data.gov.sg API)
│   └── retrieval/
│       ├── baseline.yaml                # Dense-only: embed_model, top_k
│       ├── hybrid.yaml                  # BM25 + dense + RRF: fetch, rrf_k, top_k
│       └── reranker.yaml                # Full pipeline: adds cross-encoder model + top_n
│
├── data/                                # All data artefacts (partially git-tracked)
│   ├── raw/                             # Downloaded files — not tracked
│   │   ├── documents/                   # PDF / HTML snapshots
│   │   └── structured/                  # Raw HDB resale API exports
│   ├── interim/
│   │   ├── extracted_text/              # Parsed .txt files — tracked in git
│   │   │   ├── budget/                  # Budget speeches 2023–2026 (4 PDFs)
│   │   │   ├── mas/                     # MAS Macroeconomic Reviews + MPS (5 PDFs)
│   │   │   ├── hdb/                     # HDB eligibility guides (2 JS-rendered HTML)
│   │   │   ├── cpf/                     # CPF guides — full lifecycle (15 HTML)
│   │   │   ├── srs/                     # IRAS SRS overview (1 HTML)
│   │   │   └── ssb/                     # MAS Savings Bonds overview (1 HTML)
│   │   └── cleaned_tables/              # Typed HDB resale CSV — not tracked
│   ├── processed/
│   │   └── chunks.jsonl                 # 500-char overlapping chunks — tracked in git
│   └── indexes/
│       └── chroma/                      # ChromaDB vector index — not tracked (rebuilt via make index)
│
├── src/                                 # All library code — import as src.*
│   ├── ingestion/
│   │   ├── documents/                   # Unstructured document pipeline
│   │   │   ├── fetchers/
│   │   │   │   ├── http_fetcher.py      # HTTP transport: returns raw bytes (PDF + HTML)
│   │   │   │   └── playwright_fetcher.py# JS-rendered pages: headless Chromium → raw HTML bytes
│   │   │   ├── extractors/
│   │   │   │   ├── pdf_extractor.py     # pdfplumber: bytes → plain text
│   │   │   │   └── html_extractor.py    # BeautifulSoup: bytes → plain text
│   │   │   └── pipeline.py             # Orchestrates: source registry → fetch → extract → save
│   │   └── structured_data/             # Structured data pipeline (Agentic RAG input)
│   │       ├── hdb_api_fetcher.py       # Paginated data.gov.sg CKAN API client
│   │       └── schemas.py               # Column definitions and expected types
│   ├── processing/
│   │   ├── clean_text.py                # Normalise whitespace, strip PDF artefacts
│   │   ├── chunker.py                   # Recursive character splitter with overlap
│   │   └── clean_hdb_resale.py          # Type coercion + derive year/month columns
│   ├── indexing/
│   │   ├── build_vector_index.py        # Embed chunks → ChromaDB (all-MiniLM-L6-v2)
│   │   └── build_keyword_index.py       # Build BM25Okapi from chunks.jsonl
│   ├── retrieval/
│   │   ├── loader.py                    # Load all components once per session (NamedTuple)
│   │   ├── dense.py                     # Baseline: vector search only
│   │   ├── bm25_retriever.py            # Keyword search via BM25Okapi
│   │   ├── hybrid.py                    # BM25 + dense fused with Reciprocal Rank Fusion
│   │   └── reranker.py                  # Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
│   ├── generation/
│   │   ├── prompts.py                   # SYSTEM_PROMPT + context block formatter
│   │   └── answer.py                    # answer() blocking + stream_answer() streaming
│   └── tools/                           # Agentic RAG tools (stubs — Phase 2)
│       ├── policy_search.py             # Wraps hybrid+rerank retrieval as agent-callable tool
│       └── hdb_resale_query.py          # Structured query over HDB resale parquet
│
├── pipelines/                           # CLI entry points — orchestrate src/* modules
│   ├── ingest_documents.py              # Fetch + extract all policy documents
│   ├── ingest_hdb_data.py               # Fetch HDB resale CSV + clean to parquet
│   ├── build_indexes.py                 # Chunk documents + build vector and keyword indexes
│   ├── run_eval.py                      # Run RAGAS evaluation
│   └── extract_local_pdf.py             # Fallback: extract text from a manually downloaded PDF
│
├── experiments/                         # Ablation study runs — one file per pipeline variant
│   ├── baseline.py                      # Dense-only retrieval (Stage 1 baseline)
│   ├── hybrid_retrieval.py              # BM25 + dense + RRF, no reranker
│   ├── with_reranker.py                 # Full pipeline: hybrid + cross-encoder rerank
│   └── compare_results.py              # Print comparison table across eval/results/
│
├── app/
│   └── main.py                          # Streamlit chat interface
│
├── eval/
│   ├── ragas_eval.py                    # Two-tier eval: local NLI + LLM-as-judge
│   ├── test_set.json                    # Curated Q&A pairs with ground truth
│   └── results/                         # Per-run eval output JSON files
│
├── Makefile                             # make fetch / index / app / baseline / hybrid / reranker
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
# 1. Clone and set up environment
git clone https://github.com/isafzh/asksg.git
cd asksg
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# 2. Add your Groq API key
echo GROQ_API_KEY=your_key_here > .env
# Get a free key at https://console.groq.com

# 3. Build the vector index (downloads ~90MB model on first run)
make index

# 4. Run the app
make app
# → opens at http://localhost:8501
```

To refresh the document corpus:
```bash
make fetch      # re-download source documents
make clean      # delete old index
make index      # re-chunk and re-embed
```

---

## Evaluation

RAG quality is measured on a hand-curated test set of 30 Q&A pairs drawn from the source documents (`eval/test_set.json`).

### Test set composition

| Domain | How many | Which questions | Rationale |
|---|---|---|---|
| CPF | 6 | Q1, Q2, Q3, Q19, Q20, Q21 | Most rule-heavy domain; tests contribution, housing, retirement, and LIFE rules |
| Budget | 5 | Q4, Q5, Q6, Q22, Q23 | Temporal disambiguation and year-specific policy announcements |
| HDB | 4 | Q7, Q8, Q17, Q18 | Core housing eligibility and grant questions; HDB also has a planned structured transaction-data modality for the agentic tool |
| MAS | 3 | Q9, Q10, Q24 | Macro forecasts and monetary policy framework; tests numeric projection retrieval and policy-mechanism explanation |
| SRS | 3 | Q11, Q12, Q13 | Smaller but distinct retirement/tax-relief source |
| SSB | 3 | Q14, Q15, Q16 | Smaller but distinct investment product source |
| Cross-domain | 6 | Q25–Q30 | Hardest set; tests synthesis across related policy domains; showcases hybrid retrieval strength |
| **Total** | **30** | | |

Cross-domain pairs:

Q25–Q27, Q29 require evidence from two corpus buckets:
- **Q25** `srs+cpf` — comparing SRS contributions vs CPF RA top-ups for tax relief
- **Q26** `ssb+srs` — using SRS funds to invest in SSBs, and how returns are taxed
- **Q27** `budget+mas` — MAS's April 2025 GDP projection vs Budget 2026's reported actual 5% outcome
- **Q29** `cpf+budget` — CPF and Budget measures for lower-income retirement adequacy

Q28, Q30 are domain-overlap questions answerable from one expected document:
- **Q28** `hdb+budget` — housing options and grants for singles buying near parents (evidence: HDB eligibility doc)
- **Q30** `cpf+hdb` — CPF OA usage when buying an HDB flat, plus refund obligation when sold (evidence: CPF housing usage doc)

### Test question schema

Each question carries structured metadata supporting three levels of retrieval evaluation:

| Level | What it checks | Schema field |
|---|---|---|
| L1 — Source | Was the right document in the top-k results? | `expected_sources[].document` |
| L2 — Evidence | Do the retrieved chunks contain the key facts verbatim? | `must_contain[]` |
| L3 — Chunk | Was the exact chunk retrieved? | deferred — `expected_chunks` added after chunking strategy is locked |

`must_contain` strings are grepped from the actual `.txt` corpus files so they match the exact character sequences the chunker indexes (e.g., tables may store `37` not `37%`; MAS range notation uses an en-dash `–` not a hyphen).

Questions are additionally tagged:

| Field | Values |
|---|---|
| `difficulty` | `standard` · `temporal` · `multi_chunk` · `cross_source` |
| `answer_type` | `numeric` · `policy_fact` · `eligibility` · `comparison` |
| `retrieval_mode` | `unstructured` (all 30 current questions) · `structured` (reserved for future HDB resale data queries) |

> **Note:** The results below were produced on the original corpus (13 documents) with 10 questions. The corpus has since been expanded to 25 documents across 6 sources, the test set expanded to 30 questions, ground truths for Q5 and Q7 have been corrected, and MAS Oct 2024 has been trimmed. A re-run of evaluation on the refreshed corpus and index is in progress.

Two-tier evaluation covering the full RAG Triad (Context Relevance, Faithfulness, Answer Relevance):

| Metric | What it measures | Method | API cost |
|---|---|---|---|
| Context Relevance | Are retrieved chunks on-topic? | Cosine similarity: query vs chunks (MiniLM) | 0 extra calls |
| Answer Similarity | Does the answer match ground truth? | Cosine similarity: answer vs ground truth (MiniLM) | 0 extra calls |
| Keyword Recall | Does the context contain key facts? | Ground-truth keyword presence in retrieved chunks | 0 extra calls |
| Faithfulness (LLM) | Is every claim grounded in the context? | LLM-as-judge structured prompt | 1 call/question |
| Answer Relevance (LLM) | Does the answer address the question? | LLM-as-judge structured prompt | 1 call/question |

Total per run: **60 Groq API calls** (30 generation + 30 judge). The original Ragas library required 1,300+ calls per run and exhausted the free-tier daily quota in a single run — this implementation achieves the same conceptual coverage at 4.6% of the cost.

### Step 1 — Top-K Curve (dense-only baseline)

The eval script accepts a `--top-k` argument. Four values were tested on the original dense-only pipeline to find the diminishing-returns point:

| Metric | K=5 | K=7 | K=9 | K=11 |
|---|---|---|---|---|
| Faithfulness (NLI entailment) | 0.4449 | 0.4550 | **0.4922** | 0.4659 |
| Answer Similarity (cosine) | 0.8413 | 0.8768 | 0.8777 | **0.8906** |
| Keyword Recall | 0.7405 | 0.7664 | 0.8453 | **0.8958** |

**K=9 is the sweet spot**: faithfulness peaks here then falls at K=11 (noise from weakly-related chunks causes the LLM to synthesise beyond what any single chunk supports). Answer similarity and keyword recall keep rising, but the faithfulness drop signals retrieval quality declining. Each extra chunk adds ~130 tokens to every Groq API call, so higher K also reduces daily query capacity on the free tier.

### Step 2 — Hybrid Pipeline Upgrade (k=9)

The k-curve experiment revealed a temporal disambiguation failure on Q5 (CDC Vouchers Budget 2025): `budget_2025_speech` was absent from the top-9 retrieved chunks because dense retrieval treats same-topic chunks from 2023/2024/2025 as semantically equivalent. The fix: **hybrid BM25 + dense + RRF + cross-encoder reranking**. BM25 keyword scoring on "2025" pulls the right document into the candidate pool; the cross-encoder reranker confirms its relevance.

| Metric | Baseline (dense-only, k=9) | Hybrid (BM25+dense+rerank, k=9) | Change |
|---|---|---|---|
| Context Relevance | 0.6326 | 0.6110 | -0.022 |
| **Answer Similarity** | 0.8761 | **0.9162** | **+0.040** |
| **Keyword Recall** | 0.8453 | **0.9285** | **+0.083** |
| Faithfulness (LLM) | 1.0000 | 0.9600 | -0.040 |
| **Answer Relevance (LLM)** | 0.9200 | **1.0000** | **+0.080** |

**Q5 (temporal disambiguation) — the specific failure fixed:**

| Metric | Baseline | Hybrid |
|---|---|---|
| Answer Similarity | 0.647 | **0.939** |
| Keyword Recall | 0.588 | **1.000** |
| LLM Answer Relevance | 0.6 | **1.0** |

**Trade-off:** Q7 (HDB PR eligibility) LLM faithfulness dropped from 1.0 to 0.6 — the hybrid candidate pool introduced a noisier chunk mix for that question. A more targeted fix (metadata filter by source when the query explicitly names a document category) would avoid this regression.

*30 hand-curated Q&A pairs across all 6 sources — Budget, CPF, HDB, MAS, SRS, SSB — including 6 cross-source questions.*

To run:
```bash
make baseline   # dense-only            → eval/results/baseline_k9.json
make hybrid     # BM25 + dense + RRF   → eval/results/hybrid_k9.json
make reranker   # hybrid + cross-encoder → eval/results/hybrid_rerank_k9.json
```

---

## Roadmap

- [x] Document ingestion pipeline (25 documents, 6 sources, ~784K chars)
- [x] Text preprocessing and chunking (500-char overlapping windows)
- [x] Embedding and ChromaDB indexing (all-MiniLM-L6-v2)
- [x] RAG pipeline (retrieval + Groq LLM)
- [x] Streamlit chat interface
- [x] Evaluation framework: two-tier (local NLI + LLM-as-judge), 20 Groq calls/run vs 1,300+ for Ragas
- [x] Top-K curve (k=5/7/9/11): faithfulness peaks at k=9, similarity/recall peak at k=11; k=9 chosen as optimal
- [x] Temporal disambiguation diagnosed: Q5 CDC Vouchers absent from baseline top-9 due to year-agnostic dense embeddings
- [x] Hybrid retrieval: BM25 + dense + RRF fixes Q5 (answer similarity 0.647 → 0.939, answer relevance 0.6 → 1.0)
- [x] Cross-encoder reranking: `ms-marco-MiniLM-L-6-v2` re-scores fused candidates before generation
- [x] Eval `--mode` flag: `baseline` vs `hybrid` for reproducible before/after comparison
- [ ] Metadata filtering: constrain dense retrieval by source/year for targeted queries (would fix Q7 regression)
- [ ] FastAPI backend + Docker
- [ ] AWS EC2 deployment
