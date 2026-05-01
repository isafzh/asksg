"""
RAG evaluation for AskSG — two tiers.

  Local metrics (zero extra API calls beyond generation):
    - NLI Faithfulness:    NLI entailment score per answer sentence vs retrieved chunks
                           (cross-encoder/nli-deberta-v3-base, local model)
    - Context Relevance:   Cosine similarity between query and retrieved chunks (MiniLM)
    - Answer Similarity:   Cosine similarity between answer and ground truth (MiniLM)
    - Keyword Recall:      Fraction of ground-truth keywords found in retrieved context

  LLM-as-judge metrics (10 extra Groq API calls — one structured prompt per question):
    - Faithfulness (LLM):    Is every answer claim grounded in the retrieved context?
    - Answer Relevance (LLM): Does the answer directly address the question asked?
    Unlike NLI, the LLM judge reads all chunks together and handles multi-chunk synthesis.
    This is the standard RAG Triad evaluation approach (RAGAS / TruLens).

Total API calls per run: 10 (generation) + 10 (LLM judge) = 20.

Why not just Ragas? Ragas (Plan A, tried first) made 1,300+ API calls per run —
multiple LLM calls per metric per claim per question. One run exhausted the entire
100k token/day Groq free-tier quota. This implementation achieves the same conceptual
coverage with a single structured JSON prompt per question.

Usage:
    python eval/ragas_eval.py                           # hybrid pipeline, top-k=9 (recommended)
    python eval/ragas_eval.py --mode baseline           # dense-only, for before/after comparison
    python eval/ragas_eval.py --top-k 9 --mode hybrid  # explicit (same as default)

Results saved to eval/results_k{top_k}_{mode}.json.
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

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from rag import load_retriever, answer, GROQ_MODEL
from sentence_transformers import SentenceTransformer, CrossEncoder

TEST_SET_FILE = Path(__file__).parent / "test_set.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "that", "this", "as", "it", "its", "not",
    "if", "when", "also", "both", "each", "more", "than", "into", "about",
    "what", "which", "who", "can", "per", "up", "how",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _keywords(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z0-9%$,]+\b", text.lower())
    return {w for w in words if len(w) > 3 and w not in STOPWORDS}


# ---------------------------------------------------------------------------
# Local metrics
# ---------------------------------------------------------------------------

def faithfulness_nli(answer_text: str, contexts: list[str], nli: CrossEncoder) -> float:
    """
    NLI entailment score: score each answer sentence against each chunk individually,
    take the max entailment across chunks per sentence, then average across sentences.

    Per-chunk scoring avoids the 512-token limit of nli-deberta-v3-base.
    Label order: [contradiction=0, entailment=1, neutral=2]
    """
    sentences = _sentences(answer_text)
    if not sentences:
        return 0.0
    entailment_idx = 1
    sentence_scores = []
    for s in sentences:
        pairs = [(ctx, s) for ctx in contexts]
        raw_scores = nli.predict(pairs, apply_softmax=True)
        max_entailment = max(float(row[entailment_idx]) for row in raw_scores)
        sentence_scores.append(max_entailment)
    return round(float(np.mean(sentence_scores)), 4)


def context_relevance(query: str, contexts: list[str], embedder: SentenceTransformer) -> float:
    """
    Average cosine similarity between the query and each retrieved chunk.
    Measures retrieval quality: are the chunks actually relevant to the question?
    High score = retriever found on-topic chunks. Low score = off-topic retrieval.
    """
    texts = [query] + contexts
    vecs = embedder.encode(texts, normalize_embeddings=True)
    query_vec = vecs[0]
    chunk_vecs = vecs[1:]
    scores = [float(np.dot(query_vec, cv)) for cv in chunk_vecs]
    return round(float(np.mean(scores)), 4)


def answer_similarity(answer_text: str, ground_truth: str, embedder: SentenceTransformer) -> float:
    """Cosine similarity between answer and ground truth embeddings."""
    vecs = embedder.encode([answer_text, ground_truth], normalize_embeddings=True)
    return round(max(0.0, float(np.dot(vecs[0], vecs[1]))), 4)


def keyword_recall(ground_truth: str, contexts: list[str]) -> float:
    """Fraction of ground-truth keywords present in the retrieved context."""
    gt_keywords = _keywords(ground_truth)
    if not gt_keywords:
        return 1.0
    context_text = " ".join(contexts).lower()
    found = sum(1 for kw in gt_keywords if kw in context_text)
    return round(found / len(gt_keywords), 4)


# ---------------------------------------------------------------------------
# LLM-as-judge (standard RAG Triad evaluation)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are evaluating a RAG (Retrieval-Augmented Generation) system.

Question asked by the user:
{question}

Context retrieved from the knowledge base:
{context}

Answer generated by the system:
{answer}

Score on two dimensions using a 1–5 scale:

faithfulness: Is every factual claim in the answer directly supported by the \
retrieved context above? Do not credit background knowledge the model may have — \
only what is explicitly present in the context.
  1 = answer contains major claims not in the context (hallucination)
  3 = most claims supported but some details added from outside context
  5 = every claim is fully traceable to the context

answer_relevance: Does the answer directly address what the question asked?
  1 = answer is off-topic or addresses a different question entirely
  3 = answer partially addresses the question but drifts or omits key aspects
  5 = answer squarely and completely addresses the question

Respond with valid JSON only — no explanation outside the JSON:
{{"faithfulness": <1-5>, "answer_relevance": <1-5>, "reasoning": "<one sentence>"}}"""


def llm_judge(question: str, answer_text: str, contexts: list[str]) -> dict:
    """
    LLM-as-judge: one Groq API call per question.
    Returns faithfulness_llm and answer_relevance_llm normalised to 0–1 (score/5).

    Unlike NLI, the LLM reads all chunks together and can credit multi-chunk synthesis.
    This is the standard approach used in RAGAS and TruLens.
    """
    context_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    prompt = _JUDGE_PROMPT.format(
        question=question,
        context=context_text,
        answer=answer_text,
    )
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
            "faithfulness_llm": round(parsed["faithfulness"] / 5, 4),
            "answer_relevance_llm": round(parsed["answer_relevance"] / 5, 4),
            "reasoning": parsed.get("reasoning", ""),
        }
    except (json.JSONDecodeError, KeyError):
        return {"faithfulness_llm": None, "answer_relevance_llm": None, "reasoning": raw}
    except Exception as e:
        return {"faithfulness_llm": None, "answer_relevance_llm": None, "reasoning": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AskSG RAG evaluation")
    parser.add_argument(
        "--top-k", type=int, default=9,
        help="Number of chunks passed to LLM (default: 9, optimal from k-curve eval)"
    )
    parser.add_argument(
        "--mode", choices=["baseline", "hybrid"], default="hybrid",
        help=(
            "baseline = dense-only retrieval, no rerank (original pipeline); "
            "hybrid   = BM25 + dense + RRF + cross-encoder rerank (upgraded pipeline)"
        ),
    )
    args = parser.parse_args()
    top_k = args.top_k
    mode = args.mode

    results_file = Path(__file__).parent / f"results_k{top_k}_{mode}.json"

    print("Loading test set...")
    test_set = json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))
    print(f"  {len(test_set)} questions\n")

    print("Loading RAG retriever...")
    model, collection, bm25, all_chunks, reranker = load_retriever()
    print(f"  {collection.count():,} chunks in index\n")

    print(f"Loading NLI model ({NLI_MODEL})...")
    nli = CrossEncoder(NLI_MODEL)
    print(f"Loading embedding model ({EMBED_MODEL})...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print()

    print(f"Running evaluation  [top-k={top_k}]  [mode={mode}]  —  20 Groq API calls total")
    print("  (10 for generation, 10 for LLM-as-judge)")
    print()

    rows = []
    total = len(test_set)
    for i, item in enumerate(test_set, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i}/{total}] {q[:70]}...")

        # Generation — 1 Groq call
        result = answer(q, model, collection, bm25, all_chunks, reranker, k=top_k, mode=mode)
        ans = result["answer"]
        contexts = [chunk["text"] for chunk in result["sources"]]

        # Local metrics — 0 API calls
        f_nli  = faithfulness_nli(ans, contexts, nli)
        ctx_r  = context_relevance(q, contexts, embedder)
        sim    = answer_similarity(ans, gt, embedder)
        recall = keyword_recall(gt, contexts)

        # LLM-as-judge — 1 Groq call (skipped gracefully if quota is exhausted)
        try:
            judge = llm_judge(q, ans, contexts)
        except Exception as e:
            print(f"    LLM judge skipped: {e}")
            judge = {"faithfulness_llm": None, "answer_relevance_llm": None, "reasoning": str(e)}
        f_llm  = judge["faithfulness_llm"]
        rel    = judge["answer_relevance_llm"]

        rows.append({
            "faithfulness_nli":    f_nli,
            "context_relevance":   ctx_r,
            "answer_similarity":   sim,
            "keyword_recall":      recall,
            "faithfulness_llm":    f_llm,
            "answer_relevance_llm": rel,
            "judge_reasoning":     judge["reasoning"],
        })

        print(f"    Local  — faithfulness(NLI)={f_nli:.3f}  ctx_relevance={ctx_r:.3f}  "
              f"similarity={sim:.3f}  recall={recall:.3f}")
        print(f"    LLM    — faithfulness={f_llm}  answer_relevance={rel}")

    print()

    def _mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    scores = {
        "faithfulness_nli":    _mean("faithfulness_nli"),
        "context_relevance":   _mean("context_relevance"),
        "answer_similarity":   _mean("answer_similarity"),
        "keyword_recall":      _mean("keyword_recall"),
        "faithfulness_llm":    _mean("faithfulness_llm"),
        "answer_relevance_llm": _mean("answer_relevance_llm"),
    }

    print("=" * 60)
    print(f"Evaluation Results  [top-k={top_k}]")
    print("=" * 60)
    print("  --- Local metrics (0 extra API calls) ---")
    print(f"  NLI Faithfulness:          {scores['faithfulness_nli']:.4f}")
    print(f"  Context Relevance:         {scores['context_relevance']:.4f}")
    print(f"  Answer Similarity:         {scores['answer_similarity']:.4f}")
    print(f"  Keyword Recall:            {scores['keyword_recall']:.4f}")
    print("  --- LLM-as-judge (10 extra API calls) ---")
    print(f"  Faithfulness (LLM):        {scores['faithfulness_llm']:.4f}")
    print(f"  Answer Relevance (LLM):    {scores['answer_relevance_llm']:.4f}")
    print("=" * 60)

    output = {
        "top_k": top_k,
        "mode": mode,
        "scores": scores,
        "per_question": rows,
        "n_questions": len(test_set),
        "nli_model": NLI_MODEL,
        "embed_model": EMBED_MODEL,
        "judge_model": GROQ_MODEL,
        "note": (
            "Local metrics: NLI faithfulness, context relevance, answer similarity, keyword recall. "
            "LLM-as-judge metrics: faithfulness and answer relevance (one Groq call per question). "
            "Total API calls: 10 generation + 10 judge = 20."
        ),
    }
    results_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved -> {results_file.relative_to(Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
