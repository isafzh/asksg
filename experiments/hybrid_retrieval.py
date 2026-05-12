"""
Experiment: hybrid retrieval — BM25 + dense vector search fused with RRF.

No cross-encoder reranker.  Isolates the contribution of hybrid fusion
over baseline dense retrieval.

Usage:
    python experiments/hybrid_retrieval.py
    python experiments/hybrid_retrieval.py --k 9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.loader import load
from src.generation.answer import answer

TEST_SET = ROOT / "eval" / "test_set.json"
RESULTS_DIR = ROOT / "eval" / "results"


def run(k: int = 9) -> None:
    print(f"=== Hybrid retrieval experiment (BM25 + dense + RRF, k={k}) ===\n")

    retriever = load()
    data = json.loads(TEST_SET.read_text(encoding="utf-8"))
    questions = data["questions"]

    results = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"[{i}/{len(questions)}] {q[:80]}...")
        result = answer(
            query=q,
            model=retriever.model,
            collection=retriever.collection,
            bm25=retriever.bm25,
            all_chunks=retriever.chunks,
            reranker=retriever.reranker,
            k=k,
            mode="hybrid",
        )
        results.append({
            "question": q,
            "answer": result["answer"],
            "contexts": [c["text"] for c in result["sources"]],
            "ground_truth": item.get("ground_truth", ""),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"hybrid_k{k}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} results -> {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=9)
    args = parser.parse_args()
    run(k=args.k)


if __name__ == "__main__":
    main()
