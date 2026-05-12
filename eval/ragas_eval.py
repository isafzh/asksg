"""
Cost-aware RAG evaluation for AskSG.

Evaluation layers
-----------------
Retrieval  (free - no extra API calls):
  hit_rate_at_k     Was the expected document in the top-K results?
  mrr_at_k          Reciprocal rank of first expected-document hit
  evidence_recall   Fraction of must_contain strings found in retrieved context
  context_relevance Cosine similarity between query and retrieved chunks (MiniLM)

Answer  (free - no extra API calls):
  answer_fact_recall  Fraction of must_contain strings found in the generated answer
  answer_similarity   Cosine similarity between answer and ground truth (MiniLM)
  faithfulness_nli    NLI entailment: answer sentences vs retrieved chunks

LLM judge  (optional - 1 Groq call per sampled question):
  faithfulness_llm      Is every answer claim grounded in the retrieved context?
  answer_relevance_llm  Does the answer directly address the question?
  Runs on a configurable sample (default 10). Use --judge-sample 0 to skip entirely.

Background: I initially evaluated the Ragas library, but it required 1,300+ LLM calls
per run and exhausted the Groq free-tier quota (100 k tokens/day) in a single run.
This is a cost-aware RAG evaluation suite inspired by the RAG triad (retrieval quality,
groundedness, answer relevance). Full run: 30 generation + 10 LLM judge = 40 API calls.

Usage:
    python pipelines/run_eval.py                        # hybrid_rerank, k=9 (default)
    python pipelines/run_eval.py --mode baseline        # dense-only
    python pipelines/run_eval.py --mode hybrid          # BM25 + dense + RRF, no reranker
    python pipelines/run_eval.py --judge-sample 0       # skip LLM judge entirely
    python pipelines/run_eval.py --judge-sample 5       # judge only 5 questions
    python pipelines/run_eval.py --retrieval-only       # no Groq calls; local models still load
    python pipelines/run_eval.py --top-k 7 --mode hybrid_rerank --judge-sample 10

Results saved to eval/results/<mode>_k<top_k>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from src.retrieval.loader import load as load_retriever
from src.retrieval.dense import retrieve_dense
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.reranker import rerank
from src.generation.answer import GROQ_MODEL, FETCH
from src.generation.prompts import SYSTEM_PROMPT, build_context
from sentence_transformers import CrossEncoder

TEST_SET_FILE = Path(__file__).parent / "test_set.json"
RESULTS_DIR   = Path(__file__).parent / "results"

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _normalize(s: str) -> str:
    """Surgical normalization for must_contain matching.

    Only normalizes the two character classes that differ between corpus text
    and test-set strings: Unicode dashes and curly apostrophes.
    Does NOT strip $, %, . or , -- those are meaningful in this domain.
    """
    s = s.lower()
    s = s.replace("\u2013", "-").replace("\u2014", " - ")   # en/em-dash
    s = s.replace("\u2018", "'").replace("\u2019", "'")     # curly apostrophes
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def hit_rate_at_k(expected_sources: list[dict], retrieved_chunks: list[dict]) -> float:
    """1.0 if any retrieved chunk's document matches any expected source, else 0.0."""
    expected_docs = {es["document"] for es in expected_sources}
    return 1.0 if any(c.get("document") in expected_docs for c in retrieved_chunks) else 0.0


def mrr_at_k(expected_sources: list[dict], retrieved_chunks: list[dict]) -> float:
    """Reciprocal rank of the first retrieved chunk matching an expected source."""
    expected_docs = {es["document"] for es in expected_sources}
    for rank, chunk in enumerate(retrieved_chunks, 1):
        if chunk.get("document") in expected_docs:
            return round(1.0 / rank, 4)
    return 0.0


def evidence_recall(must_contain: list[str], contexts: list[str]) -> float:
    """Fraction of must_contain strings found in retrieved context (normalized match)."""
    if not must_contain:
        return 1.0
    context_norm = _normalize(" ".join(contexts))
    found = sum(1 for needle in must_contain if _normalize(needle) in context_norm)
    return round(found / len(must_contain), 4)


def context_relevance(query: str, contexts: list[str], embedder: SentenceTransformer) -> float:
    """Average cosine similarity between the query and each retrieved chunk."""
    texts = [query] + contexts
    vecs = embedder.encode(texts, normalize_embeddings=True)
    scores = [float(np.dot(vecs[0], cv)) for cv in vecs[1:]]
    return round(float(np.mean(scores)), 4)


# ---------------------------------------------------------------------------
# Answer metrics
# ---------------------------------------------------------------------------

def answer_fact_recall(must_contain: list[str], answer_text: str) -> float:
    """Fraction of must_contain strings present in the generated answer."""
    if not must_contain:
        return 1.0
    answer_norm = _normalize(answer_text)
    found = sum(1 for needle in must_contain if _normalize(needle) in answer_norm)
    return round(found / len(must_contain), 4)


def answer_similarity(answer_text: str, ground_truth: str, embedder: SentenceTransformer) -> float:
    """Cosine similarity between answer and ground truth embeddings."""
    vecs = embedder.encode([answer_text, ground_truth], normalize_embeddings=True)
    return round(max(0.0, float(np.dot(vecs[0], vecs[1]))), 4)


def faithfulness_nli(answer_text: str, contexts: list[str], nli: CrossEncoder) -> float:
    """
    NLI entailment: score each answer sentence against each chunk individually,
    take max entailment per sentence, then average across sentences.
    Per-chunk scoring stays within the 512-token limit of nli-deberta-v3-base.
    Label order: [contradiction=0, entailment=1, neutral=2]
    """
    sentences = _sentences(answer_text)
    if not sentences:
        return 0.0
    entailment_idx = 1
    sentence_scores = []
    for s in sentences:
        pairs = [(ctx, s) for ctx in contexts]
        raw = nli.predict(pairs, apply_softmax=True)
        sentence_scores.append(max(float(row[entailment_idx]) for row in raw))
    return round(float(np.mean(sentence_scores)), 4)


# ---------------------------------------------------------------------------
# LLM judge (sampled)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are evaluating a RAG (Retrieval-Augmented Generation) system.

Question asked by the user:
{question}

Context retrieved from the knowledge base:
{context}

Answer generated by the system:
{answer}

Score on two dimensions using a 1-5 scale:

faithfulness: Is every factual claim in the answer directly supported by the \
retrieved context above? Do not credit background knowledge the model may have - \
only what is explicitly present in the context.
  1 = answer contains major claims not in the context (hallucination)
  3 = most claims supported but some details added from outside context
  5 = every claim is fully traceable to the context

answer_relevance: Does the answer directly address what the question asked?
  1 = answer is off-topic or addresses a different question entirely
  3 = answer partially addresses the question but drifts or omits key aspects
  5 = answer squarely and completely addresses the question

Respond with valid JSON only - no explanation outside the JSON:
{{"faithfulness": <1-5>, "answer_relevance": <1-5>, "reasoning": "<one sentence>"}}"""


def llm_judge(question: str, answer_text: str, contexts: list[str]) -> dict:
    """One Groq API call. Returns faithfulness_llm and answer_relevance_llm on 0-1 scale."""
    context_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    prompt = _JUDGE_PROMPT.format(question=question, context=context_text, answer=answer_text)
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        return {
            "faithfulness_llm":      round(parsed["faithfulness"] / 5, 4),
            "answer_relevance_llm":  round(parsed["answer_relevance"] / 5, 4),
            "judge_reasoning":       parsed.get("reasoning", ""),
        }
    except (json.JSONDecodeError, KeyError):
        return {"faithfulness_llm": None, "answer_relevance_llm": None, "judge_reasoning": raw}
    except Exception as e:
        return {"faithfulness_llm": None, "answer_relevance_llm": None, "judge_reasoning": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AskSG cost-aware RAG evaluation")
    parser.add_argument(
        "--top-k", type=int, default=9,
        help="Chunks passed to LLM (default 9)",
    )
    parser.add_argument(
        "--mode", choices=["baseline", "hybrid", "hybrid_rerank"], default="hybrid_rerank",
        help=(
            "baseline      = dense-only retrieval, no reranker\n"
            "hybrid        = BM25 + dense + RRF, no reranker\n"
            "hybrid_rerank = BM25 + dense + RRF + cross-encoder rerank (default)"
        ),
    )
    parser.add_argument(
        "--judge-sample", type=int, default=10,
        help="Number of questions to run through LLM judge (default 10, set 0 to skip)",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Skip generation and answer metrics; compute only retrieval metrics. No Groq API calls (local models still required).",
    )
    args = parser.parse_args()
    top_k          = args.top_k
    mode           = args.mode
    judge_sample   = args.judge_sample
    retrieval_only = args.retrieval_only

    if retrieval_only:
        judge_sample = 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix       = "_retrieval_only" if retrieval_only else ""
    results_file = RESULTS_DIR / f"{mode}_k{top_k}{suffix}.json"
    partial_file = results_file.with_suffix(".partial.json")

    print("Loading test set...")
    data      = json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))
    questions = data["questions"]
    print(f"  {len(questions)} questions  (schema {data.get('schema_version', '?')})\n")

    print("Loading retriever...")
    retriever = load_retriever()
    print(f"  {retriever.collection.count():,} chunks in index\n")

    if not retrieval_only:
        print(f"Loading NLI model ({NLI_MODEL})...")
        nli = CrossEncoder(NLI_MODEL)
    else:
        nli = None

    embedder = retriever.model  # reuse model already loaded by loader
    print()

    judge_calls = min(judge_sample, len(questions))
    if retrieval_only:
        print(f"Running retrieval-only eval  [mode={mode}]  [top-k={top_k}]  (no Groq calls; local models required)\n")
    else:
        gen_calls = len(questions)
        print(
            f"Running evaluation  [mode={mode}]  [top-k={top_k}]  "
            f"[judge-sample={judge_calls}]\n"
            f"  API calls: {gen_calls} generation + {judge_calls} LLM judge = "
            f"{gen_calls + judge_calls} total\n"
        )

    rows = []
    total = len(questions)
    for i, item in enumerate(questions, 1):
        q               = item["question"]
        gt              = item["ground_truth"]
        expected_srcs   = item.get("expected_sources", [])
        must_contain    = item.get("must_contain", [])

        print(f"  [{i:02d}/{total}] {q[:72]}...")

        # Retrieval (always runs)
        if mode == "baseline":
            chunks = retrieve_dense(q, retriever.model, retriever.collection, k=top_k)
        elif mode == "hybrid":
            chunks = retrieve_hybrid(q, retriever.model, retriever.collection,
                                     retriever.bm25, retriever.chunks, k=top_k, fetch=FETCH)
        else:  # hybrid_rerank
            chunks = retrieve_hybrid(q, retriever.model, retriever.collection,
                                     retriever.bm25, retriever.chunks, k=FETCH, fetch=FETCH)
            chunks = rerank(q, chunks, retriever.reranker, top_n=top_k)

        contexts = [c["text"] for c in chunks]

        # Retrieval metrics - 0 API calls
        hit    = hit_rate_at_k(expected_srcs, chunks)
        mrr    = mrr_at_k(expected_srcs, chunks)
        evid_r = evidence_recall(must_contain, contexts)
        ctx_r  = context_relevance(q, contexts, embedder)

        print(
            f"    Retrieval: hit={hit:.1f}  mrr={mrr:.3f}  "
            f"evidence_recall={evid_r:.3f}  ctx_relevance={ctx_r:.3f}"
        )

        ans    = None
        fact_r = None
        sim    = None
        f_nli  = None
        judge_result: dict = {"faithfulness_llm": None, "answer_relevance_llm": None, "judge_reasoning": ""}

        if not retrieval_only:
            # Generation - 1 Groq call, uses already-retrieved chunks
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {q}"},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            ans = response.choices[0].message.content

            # Answer metrics - 0 API calls
            fact_r = answer_fact_recall(must_contain, ans)
            sim    = answer_similarity(ans, gt, embedder)
            f_nli  = faithfulness_nli(ans, contexts, nli)

            print(
                f"    Answer:    fact_recall={fact_r:.3f}  "
                f"similarity={sim:.3f}  faithfulness(NLI)={f_nli:.3f}"
            )

            # LLM judge - 1 Groq call (sampled questions only)
            if i <= judge_sample:
                try:
                    judge_result = llm_judge(q, ans, contexts)
                    print(
                        f"    Judge:     faithfulness={judge_result['faithfulness_llm']}  "
                        f"answer_relevance={judge_result['answer_relevance_llm']}"
                    )
                except Exception as e:
                    print(f"    Judge:     skipped: {e}")
                    judge_result["judge_reasoning"] = str(e)

        rows.append({
            # question metadata
            "id":                    item.get("id", ""),
            "domain":                item.get("domain", ""),
            "category":              item.get("category", ""),
            "difficulty":            item.get("difficulty", ""),
            "answer_type":           item.get("answer_type", ""),
            "retrieval_mode":        item.get("retrieval_mode", ""),
            "expected_sources":      expected_srcs,
            "must_contain":          must_contain,
            # retrieved documents (for manual analysis)
            "retrieved_documents":   [c.get("document", "") for c in chunks],
            # question and answer text
            "question":              q,
            "generated_answer":      ans,
            # retrieval metrics
            "hit_rate_at_k":         hit,
            "mrr_at_k":              mrr,
            "evidence_recall":       evid_r,
            "context_relevance":     ctx_r,
            # answer metrics
            "answer_fact_recall":    fact_r,
            "answer_similarity":     sim,
            "faithfulness_nli":      f_nli,
            # judge metrics
            "faithfulness_llm":      judge_result["faithfulness_llm"],
            "answer_relevance_llm":  judge_result["answer_relevance_llm"],
            "judge_reasoning":       judge_result["judge_reasoning"],
        })

        # Incremental save - protects against mid-run quota failures
        partial_output = {
            "status":         "partial",
            "schema_version": "2.0",
            "top_k":          top_k,
            "mode":           mode,
            "retrieval_only": retrieval_only,
            "n_completed":    i,
            "n_total":        total,
            "per_question":   rows,
        }
        partial_file.write_text(json.dumps(partial_output, indent=2), encoding="utf-8")

    print()

    def _mean(key: str):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    scores = {
        "hit_rate_at_k":        _mean("hit_rate_at_k"),
        "mrr_at_k":             _mean("mrr_at_k"),
        "evidence_recall":      _mean("evidence_recall"),
        "context_relevance":    _mean("context_relevance"),
        "answer_fact_recall":   _mean("answer_fact_recall"),
        "answer_similarity":    _mean("answer_similarity"),
        "faithfulness_nli":     _mean("faithfulness_nli"),
        "faithfulness_llm":     _mean("faithfulness_llm"),
        "answer_relevance_llm": _mean("answer_relevance_llm"),
    }

    def _fmt(v) -> str:
        return f"{v:.4f}" if v is not None else "n/a"

    print("=" * 60)
    print(f"Evaluation Results  [mode={mode}]  [top-k={top_k}]")
    print("=" * 60)
    print("  --- Retrieval (0 extra API calls) ---")
    print(f"  Hit Rate@{top_k}:           {_fmt(scores['hit_rate_at_k'])}")
    print(f"  MRR@{top_k}:                {_fmt(scores['mrr_at_k'])}")
    print(f"  Evidence Recall:         {_fmt(scores['evidence_recall'])}")
    print(f"  Context Relevance:       {_fmt(scores['context_relevance'])}")
    if not retrieval_only:
        print("  --- Answer (0 extra API calls) ---")
        print(f"  Answer Fact Recall:      {_fmt(scores['answer_fact_recall'])}")
        print(f"  Answer Similarity:       {_fmt(scores['answer_similarity'])}")
        print(f"  Faithfulness (NLI):      {_fmt(scores['faithfulness_nli'])}")
        if judge_sample > 0:
            print(f"  --- LLM judge (sample n={judge_calls}) ---")
            print(f"  Faithfulness (LLM):      {_fmt(scores['faithfulness_llm'])}")
            print(f"  Answer Relevance (LLM):  {_fmt(scores['answer_relevance_llm'])}")
    print("=" * 60)

    output = {
        "schema_version":  "2.0",
        "top_k":           top_k,
        "mode":            mode,
        "retrieval_only":  retrieval_only,
        "judge_sample":    judge_calls,
        "n_questions":     len(questions),
        "scores":         scores,
        "per_question":   rows,
        "models": {
            "embed":   EMBED_MODEL,
            "nli":     NLI_MODEL,
            "judge":   GROQ_MODEL,
        },
    }
    results_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if partial_file.exists():
        partial_file.unlink()
    print(f"\nResults saved -> {results_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
