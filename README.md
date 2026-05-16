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

AskSG draws from 25 official Singapore government documents across 6 source types, totalling ~762K characters of extracted policy text.

| Source | Documents | Format | Topics covered |
|---|---|---|---|
| [Singapore Budget](https://www.singaporebudget.gov.sg) | Budget speeches 2023, 2024, 2025, 2026 | PDF | Fiscal policy, grants, cost-of-living transfers, CPF and HDB enhancements |
| [MAS](https://www.mas.gov.sg) | Macroeconomic Reviews Oct 2024, Apr 2025 (each including the Monetary Policy Statement) | PDF | Monetary policy statements, GDP and inflation forecasts, labour market, exchange rate policy |
| [HDB](https://www.hdb.gov.sg) | Eligibility guides — couples & families; singles | JS-rendered HTML | BTO/resale eligibility rules, CPF housing grants, income ceilings, SPR household rules |
| [CPF Board](https://www.cpf.gov.sg) | Contribution rates, housing usage, OW ceiling, Budget 2023 highlights, retirement sums (BRS/FRS/ERS), milestones at 55 and 65, CPF LIFE, retirement withdrawals, CPFIS options, Medisave usage, Medisave BHS, interest rates, retirement top-up scheme, top-up tax relief | HTML | Full CPF lifecycle — contributions, housing, retirement planning, Medisave, interest rates, investment scheme |
| [IRAS](https://www.iras.gov.sg) | SRS tax relief overview | HTML | Supplementary Retirement Scheme — contribution limits, tax relief, withdrawal rules |
| [MAS](https://www.mas.gov.sg) | Singapore Savings Bonds overview | HTML | SSB features, step-up interest structure, application and redemption rules |

All documents are official Singapore government publications, freely accessible to the public.

### Why these sources?

**Budget speeches (2023–2026):** The annual Budget is the primary vehicle for fiscal policy announcements — grants, CPF rule changes, HDB grant amounts, cost-of-living transfers, and tax adjustments. Four consecutive years gives the system temporal context so it can distinguish what was announced in which year, and answer comparison questions across budgets.

**MAS Macroeconomic Reviews:** Published twice yearly, each review bundles the Monetary Policy Statement (MPS) with MAS's full assessment of Singapore's growth, inflation, labour market, and external outlook. Apr 2025 is the most recent and contains 2025 growth and inflation forecasts. Oct 2024 provides the prior half-year policy context and baseline.

> [!NOTE]
> **MAS special features trimming** — This is the only document in the corpus that was partially extracted rather than used in full. The Oct 2024 review is a 100+ page academic publication. Only the main policy chapters (MPS, economic outlook, inflation, monetary and fiscal policy, data boxes) were kept. The three academic Special Feature articles — *Proceedings of the 2024 Asian Monetary Policy Forum*, *A Perspective on Inflation Targeting*, and *Globalisation is Not Dying, It's Just Changing* — were removed. These articles add ~69K chars of off-topic research content that would compete with policy chunks in retrieval without answering any realistic user query.

**HDB eligibility guides:** HDB's website is JavaScript-rendered and required Playwright for content extraction. The couples & families page (59K chars) covers eligibility by household type, CPF housing grants (Enhanced, Proximity, Step-Up), income ceilings, and the SPR 3-year rule for SPR-only households buying resale flats. The singles page (21K chars) covers the Single Singapore Citizen scheme. *HDB redesigned their site in 2025 — old URL paths now return a "We have moved" page; ETL uses the current `/flat-grant-and-loan-eligibility/` paths.*

**CPF guides (15 pages):** CPF is the most common topic for Singapore personal finance questions. Coverage spans the full CPF lifecycle: contribution rates by age band, OA savings for housing purchases, retirement sum milestones (BRS $106,500 / FRS $213,000 / ERS $426,000 for 2025), CPF LIFE monthly payouts, what happens at 55 and 65, retirement withdrawal rules, CPFIS for investing OA/SA savings, Medisave usage and the Basic Healthcare Sum ($79,000 for 2026), current interest rates (OA 2.5% / SA-MA-RA 4%), and the Retirement Top-Up Scheme (up to $16,000 tax relief per year).

**SRS (IRAS):** The Supplementary Retirement Scheme lets members voluntarily top up savings for additional tax relief — up to $15,300/year for Singapore Citizens and PRs, $35,700 for foreigners. Covers contribution caps, eligible SRS investments, and the withdrawal rules at retirement age (63).

**Singapore Savings Bonds (MAS):** Risk-free, flexible government bonds with step-up interest rates backed by the Singapore Government. The MAS page explains the application process, the $200,000 individual limit, and no-penalty early redemption — a common question from risk-averse savers.

### Corpus composition

| Source | Files | Chars | Share | Period covered |
|---|---|---|---|---|
| Budget | 4 | 336,610 | 44.1% | FY2023 – FY2026 (4 annual speeches) |
| MAS | 2 | 240,048 | 31.5% | Oct 2024 – Apr 2025 (2 half-yearly reviews, each including the MPS) |
| CPF | 15 | 89,502 | 11.7% | 2023 – 2026 (salary ceiling from Budget 2023; interest rates, BHS current as of 2026) |
| HDB | 2 | 79,836 | 10.5% | Current as of May 2026 |
| SRS | 1 | 9,185 | 1.2% | Current as of May 2026 |
| SSB | 1 | 6,976 | 0.9% | Current as of May 2026 |
| **Total** | **25** | **762,118** | | |

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
│   │   ├── reranker.py                  # Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
│   │   └── metadata_filter.py          # Dense-side source/year filter for targeted queries
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
│   ├── run_eval.py                      # Run RAG evaluation (hit rate, MRR, evidence recall, answer metrics)
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
# Windows/OneDrive: set this BEFORE building — Chroma's SQLite fails under OneDrive sync
$env:ASKSG_CHROMA_DIR = "$env:LOCALAPPDATA\AskSG\chroma"   # PowerShell (run once, or: setx ASKSG_CHROMA_DIR ...)
make index
# Windows without make: py pipelines\build_indexes.py

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
| Cross-domain | 6 | Q25–Q30 | Cross-domain topics spanning two policy areas; four require evidence from two domains, while two are domain-overlap questions answerable from one expected document |
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

`must_contain` strings are grepped from the actual `.txt` corpus files so they match the exact character sequences the chunker indexes (e.g., tables may store `37` not `37%`; strings are stored as ASCII so range notation uses a hyphen `-` not an en-dash).

Questions are additionally tagged:

| Field | Values |
|---|---|
| `difficulty` | `standard` · `temporal` · `multi_chunk` · `cross_domain` |
| `answer_type` | `numeric` · `policy_fact` · `eligibility` · `comparison` |
| `retrieval_mode` | `unstructured` (all 30 current questions) · `structured` (reserved for future HDB resale data queries) |

`domain` describes the policy area(s) a question touches. `difficulty` describes what makes retrieval hard. Therefore, a question can be cross-domain by topic but not have `difficulty: cross_domain` if the required evidence is contained in one expected document.

I initially evaluated the Ragas library, but it required 1,300+ LLM calls per run and exhausted the Groq free-tier quota (100k tokens/day) in a single run. This is a cost-aware RAG evaluation suite inspired by the RAG triad: retrieval quality, groundedness, and answer relevance.

| Layer | Metric | What it measures | Method | API cost |
|---|---|---|---|---|
| Retrieval | Hit Rate@K | Was the expected document in the top-K results? | `expected_sources` match vs chunk metadata | 0 extra calls |
| Retrieval | MRR@K | How high was the first correct source ranked? | Reciprocal rank of first hit | 0 extra calls |
| Retrieval | Evidence Recall | Do retrieved chunks contain the required facts? | `must_contain` strings present in context | 0 extra calls |
| Retrieval | Context Relevance | Are retrieved chunks on-topic? | Cosine similarity: query vs chunks (MiniLM) | 0 extra calls |
| Answer | Answer Fact Recall | Does the answer state the required facts? | `must_contain` strings present in answer | 0 extra calls |
| Answer | Answer Similarity | Does the answer match ground truth semantically? | Cosine similarity: answer vs ground truth (MiniLM) | 0 extra calls |
| Answer | Faithfulness (NLI) | Is every answer claim entailed by the context? | NLI cross-encoder (nli-deberta-v3-base, local) | 0 extra calls |
| Answer | Faithfulness (LLM) | Is every claim grounded? (multi-chunk synthesis) | LLM-as-judge structured prompt | 1 call/question |
| Answer | Answer Relevance (LLM) | Does the answer address the question? | LLM-as-judge structured prompt | 1 call/question |

Total per full run: **40 Groq API calls** (30 generation + 10 LLM judge on a fixed balanced sample covering all 6 source domains, all 4 difficulty levels, and all 4 answer types). The original Ragas library required 1,300+ calls per run — this implementation achieves the same conceptual coverage at 3% of the cost. Use `--judge-sample 0` to skip the LLM judge entirely for free iteration.

### Baseline Benchmark — Dense-only retrieval (30 questions)

Dense-only retrieval establishes a reference point and reveals failure modes. It is not a construction stage of the final system — it exists to prove the final pipeline improves something measurable.

| Metric | k=5 | k=7 |
|---|---|---|
| Hit Rate@K | 0.8333 | 0.9000 |
| MRR@K | 0.6917 | 0.7020 |
| Evidence Recall | 0.5278 | 0.6111 |
| Answer Similarity | 0.8082 | 0.8284 |
| Faithfulness (NLI) | 0.4015 | 0.4403 |

Hit rate and evidence recall improve with k, but MRR grows only slightly — the right document enters the retrieved set but is not consistently ranked first. Slow MRR growth signals a ranking weakness that expanding k alone cannot fix: dense embeddings treat year-agnostic same-topic chunks as equally relevant regardless of the queried year.

**Best completed baseline setting: k=7** — used as the comparison point for the retrieval upgrade below.

### Retrieval Upgrade — Hybrid search and reranking (30 questions)

**Motivation:** baseline MRR is 0.70 at k=7, and temporal disambiguation fails — dense retrieval ranks Budget 2023/2024/2025 chunks as equally relevant for year-specific queries. Fix: BM25 keyword scoring pulls year-specific documents into the candidate pool via RRF fusion; the cross-encoder reranker optionally re-scores the fused candidates before generation.

**Why k-sweep here?** In the hybrid+rerank pipeline, the initial candidate pool is fixed (top-25 from RRF fusion). `top_k` controls how many *post-rerank chunks* are passed to the LLM — a context-size decision, not a retrieval-depth decision. A retrieval-only k-sweep selects the optimal context window without burning Groq quota.

**Retrieval-only k-sweep (0 Groq calls):**

| k | Hit Rate | MRR | Evidence Recall | Context Relevance |
|---|---|---|---|---|
| 5 | 0.9667 | 0.8033 | 0.6389 | 0.6392 |
| 7 | 0.9667 | 0.8033 | 0.6889 | 0.6246 |
| 9 | 0.9667 | 0.8033 | 0.7389 | 0.6132 |

Hit Rate and MRR are flat across all k — the reranker consistently surfaces the right document regardless of pool size. Evidence recall improves +5pp per step; context relevance falls slightly as less-semantically-tight supporting chunks are added. **k=9 chosen**: maximises evidence recall (73.9%) with an acceptable context relevance trade-off.

With dense-side metadata filtering subsequently added (constraining Chroma dense retrieval by source/year for targeted queries), k=9 further improves: Hit Rate 1.0000 / MRR 0.8283 / Evidence Recall 0.7722 / Context Relevance 0.6075.

**Reranker ablation — hybrid-only vs hybrid+reranker at k=9 (retrieval metrics):**

| Metric | Hybrid (no reranker) | Hybrid + Reranker | Δ |
|---|---|---|---|
| Hit Rate@9 | 0.9667 | 0.9667 | 0.0 |
| MRR@9 | 0.8025 | 0.8033 | +0.0008 |
| Evidence Recall | 0.7389 | 0.7389 | 0.0 |
| Context Relevance | 0.6154 | 0.6134 | –0.002 |

**The reranker adds no measurable retrieval benefit on this corpus at k=9.** Hit Rate and Evidence Recall are identical; MRR difference (0.0008) is within noise. The reranker is an optional precision layer — it is included in the final pipeline as a robustness measure but its contribution on these 30 questions is negligible. On larger or more ambiguous corpora the benefit would likely be more pronounced.

**Final result — hybrid_rerank+filter k=9 (30 questions):**

| Layer | Metric | Best dense baseline (k=7) | Hybrid+Rerank+Filter k=9 | Δ |
|---|---|---|---|---|
| Retrieval | Hit Rate@K | 0.9000 | **1.0000** | +10.0pp |
| Retrieval | MRR@K | 0.7020 | **0.8283** | +12.6pp |
| Retrieval | Evidence Recall | 0.6111 | **0.7722** | +16.1pp |
| Retrieval | Context Relevance | — | 0.6075 | — |
| Answer | Answer Fact Recall | — | 0.5944 | — |
| Answer | Answer Similarity | 0.8284 | 0.8471 | +1.9pp |
| Answer | Faithfulness (NLI) | 0.4403 | 0.3671 | –7.3pp |
| Judge (n=10) | Faithfulness (LLM) | — | **0.9600** | — |
| Judge (n=10) | Answer Relevance (LLM) | — | **0.9200** | — |

*Metrics from `hybrid_rerank_k9.json`, the final post-filter judged run over all 30 questions.*

NLI faithfulness (0.37) lags the LLM judge (0.96) — the NLI cross-encoder penalises paraphrase even when the claim is factually supported. The LLM judge is the more reliable faithfulness signal.

The sampled LLM judge is used as a grounding/relevance diagnostic, not as the sole correctness metric. Q27 shows why: the answer can be faithful to retrieved context while still missing required benchmark facts — which is captured by `evidence_recall` and `answer_fact_recall`, not by the judge. This is intentional: the metrics cover different failure modes.

**Q22 partial improvement:** Metadata filtering retrieves the Budget 2026 source (hit=1.0, mrr=0.200), but only one of the two required facts appears in the top-9 context (evidence_recall=0.5). The second Budget 2026 chunk ranks below position 9 after reranking.

**Q27 chunk-level evidence failure:** The expected document is retrieved (hit=1.0, mrr=1.0), but the required evidence string is absent from the top-9 chunks (evidence_recall=0.0, answer_fact_recall=0.0). The LLM judge still scores faithfulness=1.0 and answer_relevance=1.0 because the generated answer is grounded in the available context and sounds relevant — it just cannot state the specific fact that was not retrieved. This is a chunk-level retrieval gap, not a corpus gap (the full document is in the index; the required sentence did not surface in the top-9).

To run:
```bash
make baseline   # dense-only benchmark              → eval/results/baseline_k7.json
make hybrid     # reranker ablation, retrieval-only  → eval/results/hybrid_k9_retrieval_only.json
make reranker   # hybrid + reranker, full judged     → eval/results/hybrid_rerank_k9.json
```

---

## Roadmap

- [x] Document ingestion pipeline (28 documents, 6 sources, ~763K chars)
- [x] Text preprocessing and chunking (500-char overlapping windows)
- [x] Embedding and ChromaDB indexing (all-MiniLM-L6-v2)
- [x] RAG pipeline (retrieval + Groq LLM)
- [x] Streamlit chat interface
- [x] Evaluation framework: retrieval (hit rate, MRR, evidence recall) + answer (fact recall, similarity, NLI) + sampled LLM judge; 40 calls/run vs 1,300+ for Ragas
- [x] **Baseline benchmark** (30 questions, dense-only): Hit Rate 0.90 / MRR 0.70 at k=7; weak MRR reveals ranking failure — dense embeddings treat year-agnostic chunks as equally relevant
- [x] Temporal disambiguation failure diagnosed: year-specific Budget queries retrieve wrong-year chunks
- [x] **Retrieval upgrade** — hybrid BM25 + dense + RRF + cross-encoder reranker: Hit Rate 0.97 / MRR 0.80 / Evidence Recall 0.74 vs dense baseline
- [x] k=9 selected via retrieval-only k-sweep (0 Groq calls): `top_k` controls post-rerank context size; evidence recall maximised at k=9 (0.7389) with flat Hit Rate/MRR
- [x] Reranker treated as optional precision layer (included in production pipeline as robustness measure)
- [x] Eval `--mode` flag: `baseline` / `hybrid` / `hybrid_rerank`; `--retrieval-only` for 0-quota k-sweeps
- [x] Hybrid-only ablation: reranker adds no measurable retrieval gain on this corpus at k=9 (Hit Rate / Evidence Recall identical; MRR Δ = 0.0008)
- [x] Dense-side metadata filtering: source/year filter constrains Chroma dense retrieval for targeted queries — Hit Rate 0.9667→1.0000, MRR +2.4pp, Evidence Recall +3.3pp; Q22 partial improvement (hit 0→1, evidence 0→0.5)
- [ ] **Agentic / multi-modal retrieval**: router choosing between policy RAG and structured HDB resale data queries
- [ ] FastAPI backend + Docker
- [ ] AWS EC2 deployment
