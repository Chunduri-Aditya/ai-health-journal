PY := python3
VENV := venv

.PHONY: setup setup-full setup-dev run test test-integration test-conversation verify verify-rag rag-eval scenario-run eval-smoke eval-smoke-retrieval report demo clean deps-check distill-behavior

setup:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements-core.txt

setup-full:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements-core.txt && pip install -r requirements-optional.txt

setup-dev:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements-dev.txt

run:
	. $(VENV)/bin/activate && $(PY) app.py

test:
	. $(VENV)/bin/activate && pytest

test-integration:
	. $(VENV)/bin/activate && pytest -m integration

verify:
	. $(VENV)/bin/activate && \
	  $(PY) -m compileall -q -x '(^|/)(archive|venv|\.git|__pycache__)/' . && \
	  pytest

verify-rag:
	. $(VENV)/bin/activate && \
	  pytest -q && \
	  PYTHONPATH=. RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) scripts/verify_rag.py

rag-eval:
	. $(VENV)/bin/activate && \
	  PYTHONPATH=. RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) evals/rag_retrieval_eval.py

eval-smoke:
	. $(VENV)/bin/activate && \
	  $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode baseline_json --mock_llm && \
	  $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode quality --mock_llm

eval-smoke-retrieval:
	. $(VENV)/bin/activate && \
	  RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode baseline_json --mock_llm && \
	  RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode quality --mock_llm

# Multi-turn conversation + RAG-memory suite. The fast tests run in the default
# `make test`; this target adds the real-Chroma `slow` memory tests.
test-conversation:
	. $(VENV)/bin/activate && \
	  pytest -q tests/test_conversation_progression.py tests/test_scenario_data.py tests/test_training_data_pipeline.py && \
	  pytest -q -m slow tests/test_conversation_memory_integration.py

# Drive every scenario through the real /analyze route + real Chroma and report
# per-turn memory recall. Regression-gated (SCENARIO_RECALL_FLOOR).
scenario-run:
	. $(VENV)/bin/activate && \
	  PYTHONPATH=. RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) evals/conversation_scenario_runner.py

report:
	. $(VENV)/bin/activate && \
	  BASE=$$(ls -t evals/results/baseline_json_*.json 2>/dev/null | head -1) && \
	  QUAL=$$(ls -t evals/results/quality_*.json 2>/dev/null | head -1) && \
	  if [ -z "$$BASE" ] || [ -z "$$QUAL" ]; then \
	    echo "Error: Missing result files. Run 'make eval-smoke' first."; \
	    exit 1; \
	  fi && \
	  $(PY) evals/summarize_results.py $$BASE $$QUAL

demo:
	. $(VENV)/bin/activate && bash tools/demo_run.sh

deps-check:
	. $(VENV)/bin/activate && pip check

distill-behavior:
	. $(VENV)/bin/activate && $(PY) tools/distill_evals_to_behavior.py

clean:
	rm -rf $(VENV) __pycache__ */__pycache__ .pytest_cache artifacts/

