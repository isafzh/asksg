"""
Lightweight RAG evaluation for AskSG — no LLM judge, zero extra API calls.

Metrics (all computed locally):
  - Faithfulness:       NLI entailment score (cross-encoder/nli-deberta-v3-base)
                        Checks whether answer sentences are supported by retrieved chunks.
  - Answer Similarity:  Cosine similarity between answer and ground truth (MiniLM embeddings)
  - Keyword Recall:     Fraction of ground-truth keywords found in retrieved context

Only the RAG generator calls Groq (1 call per question, 10 total).

Usage:
    python eval/ragas_eval.py              # default top-k=5
    python eval/ragas_eval.py --top-k 7   # retrieve 7 chunks instead

Results are printed to stdout and saved to eval/results_k{top_k}.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from dotenv import load_dotenv
load_dotenv()

from rag import load_retriever, answer
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


def _sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? — keep non-empty."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _keywords(text: str) -> set[str]:
    """Lowercase content words longer than 3 chars, not stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9%$,]+\b", text.lower())
    return {w for w in words if len(w) > 3 and w not in STOPWORDS}


def faithfulness_score(answer_text: str, contexts: list[str], nli: CrossEncoder) -> float:
    """
    NLI entailment score: for each answer sentence, score against each retrieved
    chunk individually and take the max (best-matching chunk). A sentence is
    considered faithful if at least one chunk entails it.

    Scoring each chunk separately avoids exceeding the model's 512-token limit,
    which would cause truncation and artificially low scores.

    NLI label order for nli-deberta-v3-base: [contradiction, entailment, neutral]
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


def answer_similarity(answer_text: str, ground_truth: str, embedder: SentenceTransformer) -> float:
    """Cosine similarity between answer and ground truth embeddings."""
    vecs = embedder.encode([answer_text, ground_truth], normalize_embeddings=True)
    similarity = float(np.dot(vecs[0], vecs[1]))
    return round(max(0.0, similarity), 4)


def keyword_recall(ground_truth: str, contexts: list[str]) -> float:
    """Fraction of ground-truth keywords present in the retrieved context."""
    gt_keywords = _keywords(ground_truth)
    if not gt_keywords:
        return 1.0
    context_text = " ".join(contexts).lower()
    found = sum(1 for kw in gt_keywords if kw in context_text)
    return round(found / len(gt_keywords), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="AskSG RAG evaluation")
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of chunks to retrieve per query (default: 5)"
    )
    args = parser.parse_args()
    top_k = args.top_k

    results_file = Path(__file__).parent / f"results_k{top_k}.json"

    print("Loading test set...")
    test_set = json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))
    print(f"  {len(test_set)} questions\n")

    print("Loading RAG retriever...")
    model, collection = load_retriever()
    print(f"  {collection.count():,} chunks in index\n")

    print(f"Loading NLI model ({NLI_MODEL})...")
    nli = CrossEncoder(NLI_MODEL)
    print(f"Loading embedding model ({EMBED_MODEL})...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print()

    print(f"Running RAG pipeline + local evaluation  [top-k={top_k}]...")
    rows = []
    total = len(test_set)
    for i, item in enumerate(test_set, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i}/{total}] {q[:70]}...")

        result = answer(q, model, collection, k=top_k)
        ans = result["answer"]
        contexts = [chunk["text"] for chunk in result["sources"]]

        f = faithfulness_score(ans, contexts, nli)
        s = answer_similarity(ans, gt, embedder)
        r = keyword_recall(gt, contexts)
        rows.append({"faithfulness": f, "answer_similarity": s, "keyword_recall": r})
        print(f"         faithfulness={f:.3f}  similarity={s:.3f}  recall={r:.3f}")

    print()
    scores = {
        "faithfulness":     round(float(np.mean([r["faithfulness"]     for r in rows])), 4),
        "answer_similarity": round(float(np.mean([r["answer_similarity"] for r in rows])), 4),
        "keyword_recall":   round(float(np.mean([r["keyword_recall"]   for r in rows])), 4),
    }

    print("=" * 52)
    print(f"Evaluation Results  [top-k={top_k}]")
    print("=" * 52)
    print(f"  Faithfulness (NLI entailment):  {scores['faithfulness']:.4f}")
    print(f"  Answer Similarity (cosine):     {scores['answer_similarity']:.4f}")
    print(f"  Keyword Recall:                 {scores['keyword_recall']:.4f}")
    print("=" * 52)

    output = {
        "top_k": top_k,
        "scores": scores,
        "per_question": rows,
        "n_questions": len(test_set),
        "nli_model": NLI_MODEL,
        "embed_model": EMBED_MODEL,
        "note": "All metrics computed locally. Only the RAG generator uses Groq (1 call/question).",
    }
    results_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved -> {results_file.relative_to(Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
