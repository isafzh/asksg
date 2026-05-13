.PHONY: fetch fetch-hdb index app eval baseline hybrid reranker compare clean help

help:
	@echo "AskSG — available targets:"
	@echo ""
	@echo "  Data pipeline:"
	@echo "    make fetch      Fetch and extract all policy documents"
	@echo "    make fetch-hdb  Fetch HDB resale transaction data"
	@echo "    make index      Chunk documents and build vector + keyword indexes"
	@echo ""
	@echo "  Experiments:"
	@echo "    make baseline   Run dense-only retrieval experiment"
	@echo "    make hybrid     Run hybrid (BM25 + dense + RRF) experiment"
	@echo "    make reranker   Run full pipeline (hybrid + cross-encoder) experiment"
	@echo "    make compare    Print comparison table of all scored results"
	@echo ""
	@echo "  Application:"
	@echo "    make app        Launch Streamlit app"
	@echo "    make eval       Run RAG evaluation (hit rate, MRR, evidence recall, answer metrics)"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean      Delete generated indexes (forces full rebuild)"

fetch:          # → data/interim/extracted_text/<source>/<name>.txt
	py pipelines/ingest_documents.py

fetch-hdb:      # → data/interim/cleaned_tables/hdb_resale.csv  +  data/processed/hdb_resale.parquet
	py pipelines/ingest_hdb_data.py

index:          # reads extracted_text/ → data/processed/chunks.jsonl  +  data/indexes/chroma/
	py pipelines/build_indexes.py

app:            # opens http://localhost:8501
	streamlit run app/main.py

eval:           # → eval/results/<mode>_k<k>.json
	py pipelines/run_eval.py

baseline:       # dense-only scored eval  → eval/results/baseline_k7.json
	py pipelines/run_eval.py --mode baseline --top-k 7 --judge-sample 0

hybrid:         # BM25 + dense + RRF scored eval  → eval/results/hybrid_k7.json
	py pipelines/run_eval.py --mode hybrid --top-k 7 --judge-sample 0

reranker:       # hybrid + cross-encoder scored eval  → eval/results/hybrid_rerank_k9.json
	py pipelines/run_eval.py --mode hybrid_rerank --top-k 9 --judge-sample 10

compare:        # reads eval/results/*.json  → prints table to stdout
	py experiments/compare_results.py

clean:          # deletes data/indexes/chroma/  (run 'make index' to rebuild)
	rm -rf data/indexes/chroma
	@echo "Vector index deleted. Run 'make index' to rebuild."
