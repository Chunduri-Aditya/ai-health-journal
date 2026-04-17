# AI Health Journal — Project Overview

> Single-document context dump of the repo as of the upgrade-planning checkpoint.
> This is a current-state snapshot, not a target-state architecture spec.
> Verified against tree snapshot `a17105b` on 2026-04-17.
> Read this before touching any upgrade doc under `docs/upgrades/`.

---

## 1. What this project is

A **privacy-first, local-only journaling assistant** built around Flask + Ollama, bundled with a **full LLM evaluation pipeline** and a **DPO (Direct Preference Optimization) fine-tuning pipeline**.

The distinguishing design choice: the product's own traffic patterns (baseline vs quality modes) are used to generate a **preference dataset** that trains a better local model. Baseline is *deliberately weaker* so DPO pairs don't starve on saturated ties.

Default posture is fully offline: Flask + Ollama on localhost, optional local Chroma vector store. Pinecone is gated behind `ALLOW_CLOUD_VECTORSTORE=true`.

Use this document for what exists today. Use [`docs/upgrades/README.md`](upgrades/README.md) and the per-track docs for proposed implementation changes.

---

## 2. High-level architecture

```
Browser UI (templates/index.html + static/style.css, vanilla JS)
          │  POST /analyze, /prompt, /transcribe
          │  GET  /session/history, /models, /ping
          ▼
Flask backend (app.py)
   ├── config.py                ─ env-driven dataclass settings
   ├── llm_client.py            ─ Ollama HTTP wrapper + JSON-schema enforcement + retries
   ├── generator_prompts.py     ─ draft prompt (grounded, hedged)
   ├── verifier_prompts.py      ─ verifier + revision prompts
   ├── rag_store.py             ─ LEGACY Chroma wrapper (still the one app.py uses)
   ├── vector_store/            ─ NEWER factory (none | chroma | pinecone); unused by app today
   ├── schemas/analysis.py      ─ pydantic AnalysisOutput (defined but not used in /analyze)
   ├── privacy/                 ─ local text cache + PII redaction (defined but not wired in)
   ├── behavior/                ─ rules.json, failure_patterns.json, few_shot.jsonl
   └── chains/insight_chain.py  ─ optional LangChain path (has broken fallback import)
          │
          ▼
Ollama (http://localhost:11434/api/generate)
   - phi3:3.8b          (generator / fallback)
   - samantha-mistral:7b (verifier / prompt-suggestion)
```

---

## 3. Request flow: `/analyze`

`/analyze` dispatches into one of three modes based on request body flags:

| Mode            | Trigger                       | Function             | Notes                                                                                         |
|-----------------|-------------------------------|----------------------|-----------------------------------------------------------------------------------------------|
| Legacy / Fast   | default (no flags)            | `_run_legacy`        | Single prompt → free-form therapeutic text. No JSON schema.                                   |
| Quality         | `quality_mode: true`          | `_run_quality_pipeline` | Draft (generator) → Verify (verifier as LLM-judge) → Revise (fallback) if rewrite required. |
| Baseline JSON   | `baseline_json_mode: true`    | `_run_baseline`      | Single-pass JSON. **Deliberately weaker** prompt, temp 0.3, fewer retries (3), less RAG.     |

Quality mode gates revision on `rewrite_required` OR `groundedness_score < cfg.groundedness_threshold` (default 0.75).

All three modes call `_store_in_rag()` + `_append_to_session()` on success today.

---

## 4. JSON reliability layers (`llm_client.py`)

1. Ollama `format` parameter set to a JSON schema (`DRAFT_JSON_SCHEMA` / `VERIFIER_JSON_SCHEMA`).
2. System prompt appends a CRITICAL "return ONLY JSON" instruction.
3. User prompt gets a strict reminder from attempt 3 onward.
4. Response goes through markdown-fence stripping.
5. Direct `json.loads` attempted first; falls back to `extract_json_substring` (first `{` to last `}`).
6. Up to `max_retries` attempts on parse failure.
7. **Missing:** post-parse schema validation (pydantic/jsonschema). The model could return `{}` and it would pass.

---

## 5. RAG (today)

- `rag_store.py::RAGStore` is what `app.py` actually uses via `get_rag_store()`.
- Storage: concatenates `"ENTRY: {entry}\n\nINSIGHT: {insight}"` into a single document, ID = timestamp, minimal metadata.
- **Known bug:** `_retrieve_context` in `app.py` L345–348 calls `rag_store.retrieve("", top_k=...)` — **empty query string** — so retrieval returns effectively arbitrary prior docs. The quality pipeline verifies against this.
- `vector_store/` (base + factory + chroma_store + pinecone_store) defines a structured `List[Dict[id, text, score, metadata]]` return type, supports namespaces, supports `VECTOR_BACKEND=none`. Currently unused by the running app.

---

## 6. Sessions & history (today)

- `app.py` L61: `app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)`.
  In dev, the key rotates every restart → sessions already silently expire.
- `session["chat"]` holds the full journal-entry + response + analysis_json list.
  Flask's default session is **cookie-backed** → will overflow the 4 KB cookie limit quickly.
- `/session/history` reads `session["chat"]`; `/session/reset` pops it. No server storage.

---

## 7. Evaluation & DPO pipeline

```
dataset.jsonl ──► run_evals.py (baseline_json) ──► evals/results/baseline_json_*.json
dataset.jsonl ──► run_evals.py (quality)       ──► evals/results/quality_*.json
                                 │
                                 ▼
                       build_dpo_dataset.py
                                 │
                                 ▼
                       train/dpo_pairs_*.jsonl   (+ .sample.jsonl preview)
                                 │
                                 ▼
                          train_dpo.py (TRL DPOTrainer + PEFT LoRA, 4-bit optional)
```

**Metrics per case:**
- `faithfulness` — verifier-as-judge `groundedness_score`.
- `answer_relevancy` — heuristic on required-field presence/length.
- `context_precision` / `context_recall` — Jaccard overlap with retrieved RAG context.
- `instruction_following` — JSON schema compliance.
- `no_invention` — scans verifier's `unsupported_claims` against each case's `must_not_invent` list.

**Pair filtering (`build_dpo_dataset.py::should_keep_pair`) requires:**
- Quality parse_failures == 0
- Quality faithfulness ≥ 0.95
- Quality no_invention == 1.00
- Quality strictly better than baseline, OR tie-broken by fewer unsupported claims / safer refusal phrasing
- Baseline-contains-forbidden-content regex guard (diagnose/prescribe/100% certain/upload to cloud)

---

## 8. Key files reference

| File                              | Role                                                  |
|-----------------------------------|-------------------------------------------------------|
| `app.py`                          | Flask routes + mode dispatcher + pipeline helpers     |
| `config.py`                       | `Config` dataclass; env-driven                        |
| `llm_client.py`                   | Ollama wrapper, JSON schemas, retries, `check_ollama_available` |
| `generator_prompts.py`            | `DRAFT_SYSTEM_PROMPT`, `get_draft_prompt`             |
| `verifier_prompts.py`             | `VERIFIER_SYSTEM_PROMPT`, `get_verifier_prompt`, `get_revision_prompt` |
| `rag_store.py`                    | Legacy Chroma wrapper (active)                        |
| `vector_store/{base,factory,chroma_store,pinecone_store}.py` | New abstraction (inactive)     |
| `schemas/analysis.py`             | `AnalysisOutput` pydantic (inactive in main path)     |
| `privacy/local_text_cache.py`     | Bounded JSONL cache (no callers)                      |
| `privacy/redact.py`               | PII redaction (no callers)                            |
| `chains/insight_chain.py`         | LangChain path; fallback imports broken               |
| `evals/run_evals.py`              | Evaluation harness                                    |
| `evals/build_dpo_dataset.py`      | Preference pair builder                               |
| `evals/debug_pair_deltas.py`      | Per-case delta diagnosis                              |
| `train/train_dpo.py`              | DPO LoRA trainer                                      |
| `templates/index.html`            | UI (vanilla JS)                                       |
| `static/style.css`                | UI styles                                             |
| `tools/{demo_run.sh,pinecone_bootstrap.py,set_model_env.sh,distill_evals_to_behavior.py}` | Dev utilities |
| `scripts/benchmark.py`            | `/analyze` latency profiler                           |
| `Makefile`                        | `setup`, `run`, `verify`, `eval-smoke`, `report`, `demo`, etc. |

---

## 9. Config knobs (`config.py`)

| Env var                          | Default                               | Purpose                                         |
|----------------------------------|---------------------------------------|-------------------------------------------------|
| `GENERATOR_MODEL`                | `phi3:3.8b`                           | Draft model                                     |
| `FALLBACK_MODEL`                 | `phi3:3.8b`                           | Revision model                                  |
| `VERIFIER_MODEL`                 | `samantha-mistral:7b`                 | Verifier / LLM-judge                            |
| `PROMPT_MODEL`                   | `samantha-mistral:7b`                 | Prompt-suggestion model                         |
| `QUALITY_MODE_DEFAULT`           | `false`                               | UI toggle default                               |
| `RETRIEVAL_ENABLED`              | `false` ⚠️ README example says `true` | Gate for RAG                                    |
| `RETRIEVAL_TOP_K`                | `3`                                   | Quality top-k (baseline uses top_k-1)           |
| `GROUNDEDNESS_THRESHOLD`         | `0.75`                                | Quality revision trigger                        |
| `VECTOR_BACKEND`                 | `none`                                | `none` \| `chroma` \| `pinecone`                |
| `ALLOW_CLOUD_VECTORSTORE`        | `false`                               | Must be true to use Pinecone                    |
| `PRIVACY_MODE`                   | `balanced`                            | (unused in main path today)                     |
| `RAG_NAMESPACE_MODE`             | `session`                             | (unused in main path today)                     |
| `RAG_NAMESPACE_FIXED`            | `ai-health-journal`                   | Chroma collection / Pinecone namespace          |
| `WHISPER_MODEL`                  | `base`                                | Transcription model size                        |
| `WHISPER_MAX_AUDIO_BYTES`        | `15728640` (15 MB)                    | Upload cap                                      |
| `LOCAL_CACHE_MAX_ITEMS`          | `2000`                                | Privacy cache cap (unused)                      |
| `LOCAL_CACHE_TTL_DAYS`           | `30`                                  | Privacy cache TTL (unused)                      |

---

## 10. Confirmed defects & drift (audit results)

| # | Defect                                                                                                | Location                                   |
|---|-------------------------------------------------------------------------------------------------------|--------------------------------------------|
| 1 | `_retrieve_context("")` — retrieval uses empty query                                                  | `app.py:347`                               |
| 2 | `app.secret_key` rotates per boot when env var unset                                                  | `app.py:61`                                |
| 3 | Cookie-backed session holds full chat payload (4 KB cookie cap)                                       | `app.py:356`                               |
| 4 | Dual retrieval systems; `vector_store/` unused                                                        | `rag_store.py` + `vector_store/`           |
| 5 | `import whisper` but `requirements-optional.txt` installs `faster-whisper`                            | `app.py:185`, `requirements-optional.txt:19` |
| 6 | 501 message tells user to `pip install openai-whisper` (wrong package)                                | `app.py:188–190`                           |
| 7 | Fake pipeline progress via `setTimeout` disconnected from backend                                     | `templates/index.html:168–169`             |
| 8 | Two timers assigned to same `stageTimer` variable — first is orphaned, cancellation can't reach it    | `templates/index.html:168–169`             |
| 9 | `chains/insight_chain.py` fallback imports `DRAFT_SYSTEM_PROMPT` from `llm_client` (wrong module)     | `chains/insight_chain.py:104`              |
| 10 | `Makefile` targets reference `--mock_llm`, `--verify-only`, `--mode langchain_chain` flags that `run_evals.py` doesn't implement | `Makefile:20, 24–26`     |
| 11 | README `.env` example says `RETRIEVAL_ENABLED=true`, code defaults to `false`                         | `README.md` vs `config.py:39`              |
| 12 | `AnalysisOutput` pydantic schema defined but never used to validate `/analyze` output                 | `schemas/analysis.py` + `app.py`           |
| 13 | `privacy/redact.py` has no callers in the main app; `privacy/local_text_cache.py` is consumed only by the gated Pinecone path in `vector_store/pinecone_store.py` | `privacy/redact.py`, `vector_store/pinecone_store.py:12,112` |
| 14 | `_run_baseline` uses `get_draft_prompt()` (strong user template) with a hardcoded `WEAKER_SYSTEM_PROMPT` — if `get_draft_prompt` tightens, baseline silently strengthens and DPO pairs starve | `app.py:288–320` |
| 15 | `schemas/analysis.py` uses mutable list defaults; harmless enough under pydantic here, but should be normalized to `default_factory=list` when schema validation is tightened | `schemas/analysis.py` |

---

## 11. External dependencies & runtime

- **Runtime:** Python ≥ 3.8, Flask, requests, python-dotenv.
- **LLM runtime:** Ollama on `localhost:11434`, models `phi3:3.8b` and `samantha-mistral:7b`.
- **Optional:** `chromadb` (local RAG), `pinecone-client` + `langchain-pinecone` + `sentence-transformers` (cloud RAG), `faster-whisper` + `python-multipart` (voice).
- **Training (separate stack):** `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets` (see `train/requirements.txt`).

---

## 12. Where the upgrade roadmap lives

See `docs/upgrades/README.md` for the index and revised delivery order. Each `docs/upgrades/NN-*.md` file is self-contained and can be handed to a fresh agent session.

| Upgrade | Title                                                      | File                                   |
|---------|------------------------------------------------------------|----------------------------------------|
| 01      | Fix retrieval grounding (use current entry as query)       | `01-retrieval-grounding.md`            |
| 02      | Replace cookie session history with SQLite persistence     | `02-sqlite-persistence.md`             |
| 03      | Unify the retrieval architecture (`vector_store/` only)    | `03-unify-retrieval.md`                |
| 04      | Repair and modernize voice transcription                   | `04-transcription.md`                  |
| 05      | Replace fake pipeline progress with real SSE streaming     | `05-sse-streaming.md`                  |
| 06      | Real test coverage + runtime schema validation             | `06-tests-schema.md`                   |
| 07      | Clean up dead / drifted paths                              | `07-cleanup.md`                        |

**Revised delivery order:** 06a (subset inside 06: schema + test floor) → 03 → 01 → 02 → 05 → 04 → 07 (rolling).

Rationale: land a minimal test floor + schema validation before any refactor so retrieval unification and the grounding fix are safe. Unify retrieval before fixing grounding so the fix doesn't get rewritten when `rag_store.py` is retired.
