"""One-off script: run Q5 through baseline pipeline at k=5,7,9,11 and print answers + sources."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from dotenv import load_dotenv
load_dotenv()

from rag import load_retriever, answer

Q5 = "What CDC Voucher amount was announced in Budget 2025 and when will it be disbursed?"
SEP = "=" * 62

print("Loading retriever...")
model, collection, bm25, all_chunks, reranker = load_retriever()
print("Done.\n")

for k in [5, 7, 9, 11]:
    print(SEP)
    print("k=%d | baseline (dense-only)" % k)
    print(SEP)
    result = answer(Q5, model, collection, bm25, all_chunks, reranker, k=k, mode="baseline")
    print("ANSWER:\n%s" % result["answer"])
    print("\nSOURCES (ranked by dense cosine similarity):")
    for i, s in enumerate(result["sources"], 1):
        print("  [%d] %-45s score=%.4f" % (i, s["document"], s["score"]))
    print()
