"""
Compare RAG evaluation results across pipeline variants.

Reads all scored JSON files from eval/results/ and prints a comparison table.
Each file must have a top-level "scores" key produced by pipelines/run_eval.py.

Usage:
    python experiments/compare_results.py
    python experiments/compare_results.py --metric hit_rate_at_k mrr_at_k evidence_recall
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "eval" / "results"
DEFAULT_METRICS = ["hit_rate_at_k", "mrr_at_k", "evidence_recall", "answer_similarity", "faithfulness_nli"]


def load_results(path: Path) -> dict | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return None
    if data.get("status") == "partial":
        return None
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", nargs="+", default=DEFAULT_METRICS)
    args = parser.parse_args()
    metrics = args.metric

    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print(f"No result files in {RESULTS_DIR.relative_to(ROOT)}")
        return

    rows = []
    for f in files:
        data = load_results(f)
        if data is None:
            continue
        scores = data.get("scores", {})
        rows.append({"name": f.stem, **{m: scores.get(m) for m in metrics}})

    if not rows:
        print("No scored result files found (files with a 'scores' key).")
        print("Run eval via: python pipelines/run_eval.py")
        return

    # Header
    col_w = 28
    metric_w = 16
    header = f"{'Experiment':<{col_w}}" + "".join(f"{m:>{metric_w}}" for m in metrics)
    print(header)
    print("-" * len(header))

    for row in rows:
        line = f"{row['name']:<{col_w}}"
        for m in metrics:
            v = row.get(m)
            line += f"{'N/A':>{metric_w}}" if v is None else f"{v:>{metric_w}.4f}"
        print(line)


if __name__ == "__main__":
    main()
