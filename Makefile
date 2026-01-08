PY := python3
VENV := venv

.PHONY: setup setup-full run verify eval-smoke report demo clean deps-check distill-behavior

setup:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements-core.txt

setup-full:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements-core.txt && pip install -r requirements-optional.txt

run:
	. $(VENV)/bin/activate && $(PY) app.py

verify:
	. $(VENV)/bin/activate && \
	  $(PY) -m compileall . && \
	  $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode quality --mock_llm --verify-only

eval-smoke:
	. $(VENV)/bin/activate && \
	  $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode baseline_json --mock_llm && \
	  $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode quality --mock_llm && \
	  $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode langchain_chain --mock_llm

eval-smoke-retrieval:
	. $(VENV)/bin/activate && \
	  RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode baseline_json --mock_llm && \
	  RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode quality --mock_llm && \
	  RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode langchain_chain --mock_llm

report:
	. $(VENV)/bin/activate && \
	  BASE=$$(ls -t artifacts/test_runs/evals/baseline_json_*.json 2>/dev/null | head -1) && \
	  QUAL=$$(ls -t artifacts/test_runs/evals/quality_*.json 2>/dev/null | head -1) && \
	  CHAIN=$$(ls -t artifacts/test_runs/evals/langchain_chain_*.json 2>/dev/null | head -1) && \
	  if [ -z "$$BASE" ] || [ -z "$$QUAL" ] || [ -z "$$CHAIN" ]; then \
	    echo "Error: Missing result files. Run 'make eval-smoke' first."; \
	    exit 1; \
	  fi && \
	  $(PY) evals/summarize_results.py $$BASE $$QUAL $$CHAIN

demo:
	. $(VENV)/bin/activate && bash tools/demo_run.sh

deps-check:
	. $(VENV)/bin/activate && pip check

distill-behavior:
	. $(VENV)/bin/activate && $(PY) tools/distill_evals_to_behavior.py

clean:
	rm -rf $(VENV) __pycache__ */__pycache__ .pytest_cache artifacts/

