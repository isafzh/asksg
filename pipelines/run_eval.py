"""
Pipeline: run RAG evaluation and save results to eval/results/.

Usage:
    python pipelines/run_eval.py                        # hybrid_rerank, k=9 (default)
    python pipelines/run_eval.py --mode baseline
    python pipelines/run_eval.py --mode hybrid
    python pipelines/run_eval.py --mode hybrid_rerank
    python pipelines/run_eval.py --top-k 7
    python pipelines/run_eval.py --judge-sample 0       # skip LLM judge
    python pipelines/run_eval.py --retrieval-only       # 0 Groq calls; hit rate/MRR/evidence recall only
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Delegate to the existing eval script with updated paths
from eval.ragas_eval import main

if __name__ == "__main__":
    main()
