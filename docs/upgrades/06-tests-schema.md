# Upgrade 06 — Real Test Coverage + Runtime Schema Validation

## Problem

Two gaps, one enabling the other:

### Gap 1: no conventional tests

There is no `tests/` tree. `make verify` is:

```17:20:Makefile
verify:
	. $(VENV)/bin/activate && \
	  $(PY) -m compileall . && \
	  $(PY) evals/run_evals.py --dataset evals/quick_tests.jsonl --mode quality --mock_llm --verify-only
```

That command can't protect refactors — and additionally references `--mock_llm` / `--verify-only` flags that **`evals/run_evals.py` does not implement**, so `make verify` fails today even on a clean tree. The same is true of `make eval-smoke` with its `--mode langchain_chain` flag. Entire targets are dead on arrival.

### Gap 2: no post-parse schema validation

`llm_client.py::json_generate` (L114) parses JSON and returns a raw `dict[str, Any]`:

```167:175:llm_client.py
            # Try direct parse first
            try:
                parsed = json.loads(response_text)
                return parsed
            except json.JSONDecodeError:
                # Try extracting JSON substring
                json_substring = extract_json_substring(response_text)
                parsed = json.loads(json_substring)
                return parsed
```

Ollama's `format=<schema>` parameter is a *best effort*, not a guarantee. A model could return `{}` or `{"summary": null}` and the caller would never know. Downstream code (`_format_insight`, DPO pair builder, verifier prompts) assumes fields exist and will fail later at surprising call sites.

Meanwhile `schemas/analysis.py` already defines `AnalysisOutput` as a proper pydantic model — and nothing uses it in the live path.

## Goal

1. A `tests/` directory with pytest covering config loading, storage, retrieval, route validation, transcription handler, pipeline steps, SSE formatting, and contract stability between baseline and quality modes.
2. Every LLM JSON output is validated against a pydantic model (`AnalysisOutput`, `VerifierVerdict`) immediately after parse. Validation failure is an **explicit, retryable error**, not a silent `{}`.
3. A `make verify` / `make test` target that actually runs and passes on the clean tree.

## Dependencies

- **Must land (subset) before:** 01, 02, 03, 05. Even a minimal test floor gates the larger refactors.
- The "subset first" scope is:
  - Pydantic validation in `llm_client.py::json_generate` + a pydantic `VerifierVerdict` model.
  - Contract tests for baseline vs quality (stable schema).
  - A fake `VectorStore` usable by later tracks.
  - A working `make test`.
- The rest lands progressively as each upgrade introduces new code.

## Plan

### A. Test framework & layout

- Add `requirements-dev.txt` (or a `pyproject` dev extra) for `pytest>=7`, `pytest-mock`, `pytest-cov`, and `freezegun`. Keep end-user installs on `requirements-core.txt` / `requirements-optional.txt`; do not force runtime users to install test tooling.
- `pyproject.toml` or `pytest.ini` to configure test discovery, markers, and coverage.
- Layout:

```
tests/
  __init__.py
  conftest.py                 # shared fixtures
  test_config.py
  test_llm_client.py          # JSON cleanup, extract_json_substring, retries, schema validation
  test_schema_validation.py   # AnalysisOutput, VerifierVerdict edge cases
  test_generator_prompts.py
  test_verifier_prompts.py
  routes/
    test_ping.py
    test_models.py
    test_analyze_validation.py
    test_analyze_modes.py     # via mocked ollama
    test_session_routes.py
    test_transcribe.py
  storage/
    test_repository.py
    test_migrations.py
  vector_store/
    test_contract.py          # same test against noop + fake + (opt-in) chroma
    fakes.py                  # InMemoryVectorStore
  pipeline/
    test_retrieval_step.py
    test_draft_step.py
    test_verify_step.py
  streaming/
    test_sse.py
    test_events_schema.py
  contracts/
    test_baseline_vs_quality_schema.py
```

### B. Markers

```ini
# pytest.ini
[pytest]
markers =
  integration: requires Ollama running
  slow: > 1s
addopts = -q --strict-markers -m "not integration and not slow"
```

Default run skips `integration` and `slow`; `make test-integration` or an explicit marker override runs them.

### C. Pydantic validation in `llm_client.py`

New signature:

```python
T = TypeVar("T", bound=BaseModel)

def json_generate(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_retries: int = 5,
    json_schema: Optional[Dict] = None,
    temperature: Optional[float] = None,
    validator_model: Optional[Type[T]] = None,
    return_model: bool = False,
) -> Union[Dict[str, Any], T]:
    ...
    # after successful json.loads:
    if validator_model is not None:
        try:
            validated = validator_model.model_validate(parsed)
            return validated if return_model else validated.model_dump()
        except ValidationError as e:
            # Surface the field-level error to the next retry attempt:
            last_validation_error = e
            continue
    return parsed
```

Error handling rule: if `validator_model` is provided and all retries exhaust, raise `ValueError("json_schema_validation_failed: <field>")` — not a bare parse error. The caller (`_run_quality_pipeline`, `_run_baseline`) can map that to the same `"json_parse_failed:stage=..."` error envelope `/analyze` already returns.

### D. Schemas

Tighten `schemas/analysis.py`:

```python
class AnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    emotions: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    coping_suggestions: List[str] = Field(default_factory=list, max_length=10)
    quotes_from_user: List[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)

    # Optional enrichment (kept from existing schema)
    grounding_evidence: List[str] = Field(default_factory=list)
    grounding_sources: List[GroundingSource] = Field(default_factory=list)
    grounding_mode: Optional[str] = None
    retrieval_top_k: Optional[int] = None
    uncertainties: List[str] = Field(default_factory=list)
```

Add new module `schemas/verifier.py`:

```python
class VerifierVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groundedness_score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: List[str] = Field(default_factory=list)
    safety_flags: List[str] = Field(default_factory=list)
    rewrite_required: bool
    rewrite_instructions: str = ""
```

Wire both in `app.py` and `evals/run_evals.py`.

### E. Fake `VectorStore`

**`tests/vector_store/fakes.py`**

```python
class InMemoryVectorStore(VectorStore):
    enabled = True
    def __init__(self):
        self._docs: Dict[str, Dict[str, List[Tuple[str, str, dict]]]] = {}
    # trivial substring-overlap scoring so tests are deterministic
    ...
```

Used by the `vector_store/test_contract.py` contract test — the same test runs against `NoOpStore`, `InMemoryVectorStore`, and optionally `ChromaStore` (when `@pytest.mark.integration`).

### F. Contract tests — baseline vs quality

`tests/contracts/test_baseline_vs_quality_schema.py` pins the DPO-critical invariants:

- Both modes return `AnalysisOutput`-valid JSON.
- Baseline uses temp = 0.3 and max_retries = 3 (grep for constants or inject via `json_generate` kwargs).
- Baseline's system prompt is a distinct, weaker string from the quality system prompt (hashes differ).
- `/analyze` response shape is stable across modes (`insight` + `analysis` present when in JSON mode; `sources` present after 01).

If someone later "improves" the baseline by making it stricter, this test fires and they know DPO pair generation is about to starve.

### G. Route tests

`tests/routes/test_analyze_modes.py` patches:

- `llm_client.ollama_generate` with canned responses keyed by call count / model name.
- `vector_store.factory.get_vector_store` with `InMemoryVectorStore`.

Asserts full route behavior end-to-end without an Ollama process.

### H. Replace broken Makefile targets

```make
.PHONY: test test-integration verify eval-smoke

test:
	. $(VENV)/bin/activate && pytest -q

test-integration:
	. $(VENV)/bin/activate && pytest -q -m integration

verify:
	. $(VENV)/bin/activate && \
	  $(PY) -m compileall -q . && \
	  pytest -q && \
	  $(PY) -m py_compile $$(git ls-files '*.py')

eval-smoke:
	# delete or rewrite; see 07-cleanup.md for disposition of removed flags
```

Final shape of `eval-smoke` is decided in 07 after the `--mock_llm` / `langchain_chain` flags are either implemented or removed.

### I. CI hook (optional, low cost)

`.github/workflows/test.yml`:

```yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements-dev.txt
      - run: pytest -q
```

## New / changed interfaces

### `llm_client.json_generate`

New keyword-only args:

- `validator_model: Type[BaseModel] | None`
- `return_model: bool = False`

When validation is enabled, retries re-attempt on `ValidationError` as well as `JSONDecodeError`. Callers can keep today's dict-shaped behavior by leaving `return_model=False`.

### Pydantic models (finalized shapes)

- `schemas.analysis.AnalysisOutput`
- `schemas.verifier.VerifierVerdict`
- `schemas.events.*` (from 05, typed SSE envelopes)

### Makefile

- `make test` — unit + fast tests (no Ollama required).
- `make test-integration` — opt-in, needs Ollama.
- `make verify` — compile + `make test`.

## Acceptance criteria

1. `make test` succeeds on a clean tree without Ollama running.
   Assumes dev dependencies are installed from `requirements-dev.txt`.
2. `pytest -q` exits non-zero when:
   - A model returns `{}` in draft mode (catches the schema gap).
   - A retrieval hit from namespace A leaks into namespace B (post-01).
   - Baseline's system prompt is changed to match quality's (DPO starvation canary).
3. Coverage report: ≥ 70 % on `app.py`, `llm_client.py`, `storage/`, `vector_store/`.
4. `grep -rn "--mock_llm\|--verify-only\|langchain_chain"` returns zero hits in `Makefile` and shell scripts — or corresponding flags are actually implemented.
5. All tracks (01–05, 07) use `AnalysisOutput` / `VerifierVerdict` in their new code paths.

## Risks & open questions

- **Pydantic version.** `schemas/analysis.py` uses pydantic v2 syntax (`Field(ge=..., le=...)`). Pin `pydantic>=2,<3` in `requirements-core.txt`. Some older dependent packages may bind v1 — check after the pin.
- **Ollama in CI.** GitHub Actions can't easily run Ollama + a 4 GB model. Keep `integration` tests opt-in; CI runs only the fast suite.
- **Test fixture drift.** Canned Ollama responses encoded in tests will drift from real model behavior. That's fine — the fast tests aren't for measuring model quality, they're for measuring plumbing. Real quality is measured by `evals/`.
- **`_run_baseline` weakness contract.** The baseline weakness is currently encoded as inline constants inside `_run_baseline`. The contract test should either read those constants directly or move them to `config.py` / a dataclass to be testable. Prefer the latter.
- **Return-shape churn.** Returning pydantic model instances directly from `json_generate` would force broad call-site edits. Keep dict-by-default semantics during the initial hardening pass.

## Touch list

- `requirements-dev.txt` — new; include `-r requirements-core.txt` plus pytest tooling.
- `pytest.ini` — new.
- `llm_client.py` — add `validator_model` kwarg; call `model_validate` on success; retry on `ValidationError`.
- `schemas/analysis.py` — tighten with `extra="forbid"` and bounds.
- `schemas/verifier.py` — new.
- `schemas/events.py` — new (SSE envelopes; under 05).
- `tests/` — all new files listed in section A.
- `Makefile` — rewrite `verify`, add `test`, `test-integration`; remove or retool dead targets.
- `.github/workflows/test.yml` — optional CI.
- `app.py`, `evals/run_evals.py` — pass `validator_model` through to `json_generate`.
