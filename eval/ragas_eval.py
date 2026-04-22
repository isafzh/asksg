"""
Ragas evaluation for AskSG RAG pipeline.

Runs 4 metrics on a hand-curated test set of 18 Q&A pairs:
  - Faithfulness:      Does the answer contain only claims supported by retrieved chunks?
  - Answer Relevancy:  Is the answer relevant to the question?
  - Context Precision: Are the retrieved chunks relevant (low noise)?
  - Context Recall:    Did retrieval find all information needed to answer?

Usage:
    python eval/ragas_eval.py

Results are printed to stdout and saved to eval/results.json.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Allow importing from app/
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from dotenv import load_dotenv
load_dotenv()

from rag import load_retriever, answer

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (  # type: ignore[attr-defined]
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper  # type: ignore[attr-defined]
from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[attr-defined]
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

TEST_SET_FILE = Path(__file__).parent / "test_set.json"
RESULTS_FILE = Path(__file__).parent / "results.json"
GROQ_MODEL = "llama-3.3-70b-versatile"
JUDGE_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL = "all-MiniLM-L6-v2"


def build_dataset(test_set: list[dict], model, collection) -> EvaluationDataset:
    samples = []
    total = len(test_set)
    for i, item in enumerate(test_set, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i}/{total}] {q[:70]}...")

        result = answer(q, model, collection)
        contexts = [chunk["text"] for chunk in result["sources"]]

        samples.append(SingleTurnSample(
            user_input=q,
            response=result["answer"],
            retrieved_contexts=contexts,
            reference=gt,
        ))

    return EvaluationDataset(samples=samples)


def main() -> None:
    print("Loading test set...")
    test_set = json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))
    print(f"  {len(test_set)} questions loaded\n")

    print("Loading RAG retriever...")
    model, collection = load_retriever()
    print(f"  {collection.count():,} chunks in index\n")

    print("Running RAG pipeline on all questions...")
    dataset = build_dataset(test_set, model, collection)
    print()

    print(f"Configuring Ragas judge LLM ({JUDGE_MODEL}) + MiniLM embeddings...")
    groq_llm = LangchainLLMWrapper(
        ChatGroq(
            model=JUDGE_MODEL,
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0,
        )
    )
    hf_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL, show_progress=False)
    )

    metrics = [
        Faithfulness(llm=groq_llm),
        AnswerRelevancy(llm=groq_llm, embeddings=hf_embeddings),
        ContextPrecision(llm=groq_llm),
        ContextRecall(llm=groq_llm),
    ]
    print()

    print("Evaluating... (this takes a few minutes — each metric calls the LLM)")
    results = evaluate(dataset=dataset, metrics=metrics)
    print()

    scores = {
        "faithfulness":      round(float(results["faithfulness"]), 4),
        "answer_relevancy":  round(float(results["answer_relevancy"]), 4),
        "context_precision": round(float(results["context_precision"]), 4),
        "context_recall":    round(float(results["context_recall"]), 4),
    }

    print("=" * 50)
    print("Ragas Evaluation Results")
    print("=" * 50)
    print(f"  Faithfulness:        {scores['faithfulness']:.4f}")
    print(f"  Answer Relevancy:    {scores['answer_relevancy']:.4f}")
    print(f"  Context Precision:   {scores['context_precision']:.4f}")
    print(f"  Context Recall:      {scores['context_recall']:.4f}")
    print("=" * 50)

    output = {"scores": scores, "n_questions": len(test_set), "generator_model": GROQ_MODEL, "judge_model": JUDGE_MODEL}
    RESULTS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved -> {RESULTS_FILE.relative_to(Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
