# AskSG

A Retrieval-Augmented Generation (RAG) assistant for Singapore public policy documents.

Ask questions in natural language about Singapore's Budget, housing policies, CPF rules, and monetary policy — and get answers grounded in official government sources.

**Live demo:** *(coming soon)*

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
    [ ETL / Ingestion ]          etl/fetch_documents.py
    Download PDFs + HTML
            │
            ▼
    [ Preprocessing ]            etl/chunk_documents.py
    Clean + split into
    overlapping text chunks
    (500 chars, 50 overlap)
            │
            ▼
    [ Embedding + Indexing ]     etl/build_index.py
    sentence-transformers
    (all-MiniLM-L6-v2)
    → ChromaDB vector store
            │
            ▼
    [ RAG Pipeline ]             app/rag.py
    User question
    → embed → retrieve top-k chunks
    → Groq LLM (Llama 3.3 70B)
    → grounded answer
            │
            ▼
    [ Streamlit Interface ]      app/main.py
    Chat interface
```

---

## Data Sources

| Source | Documents | Format |
|---|---|---|
| [Singapore Budget](https://www.singaporebudget.gov.sg) | Budget speeches 2023, 2024, 2025 | PDF |
| [MAS](https://www.mas.gov.sg) | Macroeconomic Reviews Apr/Oct 2024, Apr 2025 | PDF |
| [HDB](https://www.hdb.gov.sg) | Buying guides, resale eligibility, couples & families | HTML |
| [CPF Board](https://www.cpf.gov.sg) | Contribution rates, housing usage, OW ceiling schedule, Budget 2023 CPF highlights | HTML |

All documents are official Singapore government publications, free for public use.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.9+ |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (local, persistent) |
| LLM | Groq API — `llama-3.3-70b-versatile` (free tier) |
| PDF parsing | `pdfplumber` |
| Frontend | Streamlit |
| Deployment *(planned)* | FastAPI + Docker + AWS EC2 |

---

## Project Structure

```
asksg/
├── corpus/                    # Extracted text documents (tracked in git)
│   ├── budget/                # Singapore Budget speeches (3 PDFs)
│   ├── mas/                   # MAS Macroeconomic Reviews (3 PDFs)
│   ├── hdb/                   # HDB eligibility guides (3 HTML)
│   ├── cpf/                   # CPF contribution and housing guides (4 HTML)
│   └── chunks.jsonl           # 2,107 text chunks ready for embedding
├── etl/
│   ├── fetch_documents.py     # Downloads source documents
│   ├── chunk_documents.py     # Cleans and chunks text
│   ├── build_index.py         # Embeds chunks into ChromaDB
│   └── extract_local_pdf.py   # Helper: extract text from local PDFs
├── app/
│   ├── rag.py                 # RAG pipeline (retrieve + Groq generate)
│   └── main.py                # Streamlit chat interface
├── eval/
│   ├── ragas_eval.py          # Evaluation script (local NLI + cosine similarity)
│   └── test_set.json          # 10 curated Q&A pairs with ground truth
├── .gitignore
└── requirements.txt
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
python etl/build_index.py

# 4. Run the app
streamlit run app/main.py
# → opens at http://localhost:8501
```

To refresh the document corpus:
```bash
python etl/fetch_documents.py   # re-download source documents
python etl/chunk_documents.py   # re-chunk
python etl/build_index.py       # re-embed (delete chroma_db/ first)
```

---

## Evaluation

RAG quality is measured on a hand-curated test set of 10 Q&A pairs drawn from the source documents (`eval/test_set.json`).

All metrics are computed locally — no extra API calls beyond the 10 needed to generate answers:

| Metric | What it measures | Method |
|---|---|---|
| **Faithfulness** | Are answer claims supported by retrieved chunks? | NLI entailment score (`cross-encoder/nli-deberta-v3-base`) |
| **Answer Similarity** | Does the answer match the ground truth? | Cosine similarity — MiniLM embeddings |
| **Keyword Recall** | Does retrieved context contain ground truth facts? | Keyword string matching |

To run:
```bash
python eval/ragas_eval.py
```

*(Evaluation scores will be published here after the first run.)*

---

## Roadmap

- [x] Document ingestion pipeline (13 documents, 4 sources)
- [x] Text preprocessing and chunking (2,107 chunks)
- [x] Embedding and ChromaDB indexing (2,107 vectors, all-MiniLM-L6-v2)
- [x] RAG pipeline (retrieval + Groq LLM)
- [x] Streamlit chat interface
- [x] Evaluation framework (local NLI + cosine similarity, 10-question test set)
- [ ] Evaluation results (pending first run)
- [ ] FastAPI backend + Docker
- [ ] AWS EC2 deployment
