PY := python3
VENV := venv

# Scratch Chroma directory for every eval and test target below.
#
# These targets used to inherit CHROMA_PERSIST_DIR's default of ./storage/chroma
# and wrote fixture text straight into the real journal store. Measured damage:
# 331 collections holding 280 entries of which only 23 texts were unique, and
# adversarial fixtures ("Sometimes I think about harming myself") surfacing as
# retrieval results for unrelated queries in the app.
#
# .runtime/ is gitignored. Nothing here is precious; delete it any time.
EVAL_CHROMA_DIR := .runtime/chroma-eval

.PHONY: setup setup-full setup-dev run test test-integration test-conversation verify verify-rag rag-eval crisis-eval reframe-eval ingest-reference scenario-run eval-smoke eval-smoke-retrieval report demo clean deps-check distill-behavior

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
	. $(VENV)/bin/activate && CHROMA_PERSIST_DIR=$(EVAL_CHROMA_DIR) pytest

test-integration:
	. $(VENV)/bin/activate && CHROMA_PERSIST_DIR=$(EVAL_CHROMA_DIR) pytest -m integration

verify:
	. $(VENV)/bin/activate && \
	  $(PY) -m compileall -q -x '(^|/)(archive|venv|\.git|__pycache__)/' . && \
	  CHROMA_PERSIST_DIR=$(EVAL_CHROMA_DIR) pytest && \
	  PYTHONPATH=. $(PY) evals/crisis_safety_eval.py

verify-rag:
	. $(VENV)/bin/activate && \
	  CHROMA_PERSIST_DIR=$(EVAL_CHROMA_DIR) pytest -q && \
	  PYTHONPATH=. CHROMA_PERSIST_DIR=$(EVAL_CHROMA_DIR) RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) scripts/verify_rag.py

# Sensitivity/specificity for the deterministic crisis floor. No LLM, no network.
# Gated: crisis sensitivity must stay at 1.0 on covered phrasing, because the
# floor exists to fail closed rather than degrade quietly.
crisis-eval:
	. $(VENV)/bin/activate && \
	  PYTHONPATH=. $(PY) evals/crisis_safety_eval.py

# Validates the reframe-quality rubric against labeled good/bad exemplars (no
# LLM). Add --live (needs Ollama) to score the real pipeline's output instead.
reframe-eval:
	. $(VENV)/bin/activate && \
	  PYTHONPATH=. $(PY) evals/reframe_quality_eval.py

# Ingests OpenStax Psychology 2e (CC BY-NC-SA 4.0) into the reference corpus
# namespace. Requires RETRIEVAL_ENABLED=true and REFERENCE_CORPUS_ENABLED=true
# to actually be used by /analyze afterward. Cached fetches and the corpus
# itself live under .runtime/ and storage/, both gitignored -- never committed.
ingest-reference:
	. $(VENV)/bin/activate && \
	  PYTHONPATH=. $(PY) scripts/ingest_reference_corpus.py

rag-eval:
	. $(VENV)/bin/activate && \
	  PYTHONPATH=. CHROMA_PERSIST_DIR=$(EVAL_CHROMA_DIR) RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) evals/rag_retrieval_eval.py

eval-smoke:
	. $(VENV)/bin/activate && \
	  $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode baseline_json --mock_llm && \
	  $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode quality --mock_llm

eval-smoke-retrieval:
	. $(VENV)/bin/activate && \
	  CHROMA_PERSIST_DIR=$(EVAL_CHROMA_DIR) RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode baseline_json --mock_llm && \
	  CHROMA_PERSIST_DIR=$(EVAL_CHROMA_DIR) RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode quality --mock_llm

# Multi-turn conversation + RAG-memory suite. The fast tests run in the default
# `make test`; this target adds the real-Chroma `slow` memory tests.
test-conversation:
	. $(VENV)/bin/activate && \
	  CHROMA_PERSIST_DIR=$(EVAL_CHROMA_DIR) pytest -q tests/test_conversation_progression.py tests/test_scenario_data.py tests/test_training_data_pipeline.py && \
	  CHROMA_PERSIST_DIR=$(EVAL_CHROMA_DIR) pytest -q -m slow tests/test_conversation_memory_integration.py

# Drive every scenario through the real /analyze route + real Chroma and report
# per-turn memory recall. Regression-gated (SCENARIO_RECALL_FLOOR).
scenario-run:
	. $(VENV)/bin/activate && \
	  PYTHONPATH=. CHROMA_PERSIST_DIR=$(EVAL_CHROMA_DIR) RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma $(PY) evals/conversation_scenario_runner.py

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
	rm -rf $(VENV) __pycache__ */__pycache__ .pytest_cache artifacts/ $(EVAL_CHROMA_DIR)

