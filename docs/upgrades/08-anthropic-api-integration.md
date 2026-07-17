# Upgrade 08 — Anthropic API Integration (provider abstraction + Claude-as-teacher)

> Self-contained task spec. Hand this whole file to a fresh **Claude Sonnet (latest, extended thinking ON)** Cowork session.
> Read `docs/PROJECT_OVERVIEW.md` first — it is the current-state snapshot. Do not trust this file over the code; verify against the tree.

---

## 0. Operating rules for the session

- Extended thinking ON. This is an architecture change across `llm_client.py`, `config.py`, `model_selection.py`, and the eval/DPO pipeline. Plan before editing.
- **Privacy-first posture is non-negotiable.** Default runtime stays fully local (Ollama). Cloud LLM is opt-in only, gated behind an explicit env flag, exactly like the existing Pinecone gate (`ALLOW_CLOUD_VECTORSTORE`). Never make Anthropic the default backend.
- Match existing code conventions: `from __future__ import annotations`, dataclass config, type hints, no new heavyweight deps beyond `anthropic`.
- Every behavior change ships with a test. The repo already has `tests/` with contract tests — extend them, do not bypass.
- Do not credit any AI agent in commits, comments, or doc footers. Commits are the user's.
- Work in delivery order (§5). Land the test + schema floor before the refactor so nothing silently regresses.

---

## 1. Goal

Add an optional Anthropic (Claude) backend to AI Health Journal without breaking the local-only default. Two surfaces:

1. **Runtime provider** — a clean `LLMProvider` interface so `app.py` can call either Ollama (default) or Claude (opt-in) for draft / verify / revise / prompt, with no change to the route logic.
2. **Offline teacher** — use Claude in the eval + DPO pipeline as (a) the gold groundedness judge and (b) the "chosen" response generator, so the local model distills from Claude. This is where the cloud call earns its keep; runtime can stay 100% local.

Both surfaces share one provider layer. Build it once.

---

## 2. Why this design (read before coding)

- The product's value prop is "privacy-first, local-only." A naive `requests.post` to Anthropic in the `/analyze` hot path destroys that. So cloud is gated and off by default.
- The DPO pipeline (`evals/build_dpo_dataset.py`) needs a strong "chosen" signal and a reliable judge. Today the judge is `samantha-mistral:7b` (weak, noisy). Claude as judge + chosen-generator turns this into proper **distillation**: phi3 learns to imitate Claude's grounded, hedged, schema-valid output. Cloud only ever sees your own eval corpus, offline, never user traffic.
- Anthropic's forced **tool-use** gives guaranteed-shape JSON. That lets the Anthropic path skip the entire `_strip_markdown_fences` → `extract_json_substring` → 5x retry ladder in `llm_client.py`. Fewer failure modes on that branch.

---

## 3. Architecture — what to build

### 3.1 Provider interface (`llm_client.py` or new `providers/`)

Introduce a minimal protocol both backends satisfy:

```python
class LLMProvider(Protocol):
    def generate(self, model: str, prompt: str, *, system: str | None = None,
                 temperature: float | None = None, timeout: int = 30) -> str: ...
    def json_generate(self, model: str, system_prompt: str, user_prompt: str, *,
                      json_schema: dict, max_retries: int = 5,
                      temperature: float | None = None,
                      validator_model: type[BaseModel] | None = None) -> dict: ...
```

- `OllamaProvider` — wraps the existing functions. Behavior-preserving; existing tests must still pass unchanged.
- `AnthropicProvider` — uses the `anthropic` SDK. For `json_generate`, force a single tool whose `input_schema` is the same JSON schema (`DRAFT_JSON_SCHEMA` / `VERIFIER_JSON_SCHEMA`) via `tool_choice={"type": "tool", "name": ...}`. Read `message.content[0].input` as the parsed dict. Skip the markdown/retry stack — forced tool use returns structured input directly. Still run `validator_model.model_validate` when supplied (see Upgrade 06 / defect #12 — schema validation must be wired in on both paths).

Map roles → models through `model_selection.py`, which already abstracts generator/fallback/verifier/prompt. Add an Anthropic role table (e.g. all roles → latest Sonnet; optionally Haiku for the lightweight `prompt` role to control cost). **Confirm the exact current model IDs in the Anthropic console before pinning** — do not hard-code a string from memory. Use the latest Claude Sonnet for generator/verifier/fallback.

### 3.2 Backend selection + gate (`config.py`)

Add:

| Env var | Default | Purpose |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` \| `anthropic` |
| `ALLOW_CLOUD_LLM` | `false` | Hard gate. If `anthropic` selected but this is false → refuse to start the cloud path, log a clear message, fall back to Ollama. Mirror `ALLOW_CLOUD_VECTORSTORE`. |
| `ANTHROPIC_API_KEY` | (unset) | Read from env only. Never log it. Never write it to session, cache, or RAG metadata. |
| `ANTHROPIC_GENERATOR_MODEL` / `_VERIFIER_MODEL` / `_PROMPT_MODEL` | latest Sonnet / Sonnet / Haiku | Per-role overrides. |

A factory `get_llm_provider(cfg)` returns the gated provider, analogous to `vector_store/factory.py::get_vector_store`. If the gate blocks cloud, return Ollama and log why.

### 3.3 Wire into `app.py`

`/analyze`, `/prompt` call `provider.generate` / `provider.json_generate` instead of the module-level Ollama functions. The three modes (legacy / quality / baseline) are unchanged in shape — only the call target moves behind the provider. `check_ollama_available()` becomes `provider.healthcheck()` so the 503 path works for either backend (Anthropic health = key present + cheap ping or just key-present check; do not burn tokens on every request).

### 3.4 Offline teacher path (the high-value surface)

In `evals/`:
- Add a Claude-backed generation target so `run_evals.py` can produce a `claude_*` result set alongside `baseline_json_*` and `quality_*`.
- In `build_dpo_dataset.py`, allow `chosen` to come from the Claude target (gated, offline, explicit flag — not the default eval run). Keep the existing `should_keep_pair` guards (faithfulness ≥ 0.95, no_invention == 1.0, forbidden-content regex). Claude output must clear the same bar — no free pass.
- Optionally add a Claude judge mode for `faithfulness` scoring, replacing the local verifier in eval (not in runtime). Document that this changes the metric's provenance in `RESULTS.md` rows (note the judge model + version).

Keep all of this **offline and opt-in**. It runs on your eval corpus, never on live user entries.

---

## 4. Defects to fix as part of this upgrade (do not expand scope beyond these)

These are already logged in `PROJECT_OVERVIEW.md §10`. Fixing them is in-scope because the provider refactor touches the same lines:

- **#12** — wire `validator_model` / pydantic `AnalysisOutput` validation into the `/analyze` output on **both** providers. `{}` must not pass.
- **#1** — while you are in `_retrieve_context`, fix the empty-query retrieval (`vector_store.query("", ...)`). Use the current journal entry as the query. (This is Upgrade 01; if 01 already landed, skip.)
- **#14** — note in code comments that the Anthropic baseline target must stay deliberately weaker if used for DPO, same as the Ollama baseline, or pairs starve.

Everything else in the defect table is out of scope for this PR.

---

## 5. Delivery order

1. **Schema-validation floor (#12)** on the existing Ollama path + a test that a `{}` response is rejected. Land first so the refactor can't regress it.
2. **Provider interface + `OllamaProvider`** as a behavior-preserving wrapper. All existing tests green, no behavior change.
3. **`AnthropicProvider`** + `config.py` gate + `get_llm_provider` factory. Unit-test the gate: `anthropic` + `ALLOW_CLOUD_LLM=false` → falls back to Ollama, logs reason, never instantiates the client.
4. **Wire `app.py`** to the provider. Manual smoke: `LLM_BACKEND=ollama` unchanged; `LLM_BACKEND=anthropic ALLOW_CLOUD_LLM=true` with a key produces valid schema-passing JSON.
5. **Offline teacher path** in `evals/` (separate, opt-in). Lowest risk, highest research value; ship last so it can't block the runtime work.

---

## 6. Acceptance criteria

- `make test` and `make lint` pass.
- Default run (no env changes) is byte-for-byte the same Ollama behavior. Privacy posture intact: no network call leaves localhost unless `ALLOW_CLOUD_LLM=true`.
- With the gate open + a key: `/analyze` quality mode returns schema-valid JSON via forced tool use, no retry-ladder log noise.
- Gate test: cloud requested but not allowed → Ollama fallback + one clear warning log. No `anthropic.Anthropic()` instantiated, no key read into memory beyond the gate check.
- `ANTHROPIC_API_KEY` appears in zero logs, zero session payloads, zero RAG metadata. Grep to prove it.
- New env vars documented in `README.md` and `.env.example`.
- DPO teacher path runs offline on the eval corpus only and respects existing `should_keep_pair` guards.

---

## 7. Out of scope (send to BACKLOG, do not build)

- Streaming Claude responses to the UI (that's Upgrade 05's surface).
- Per-user cloud key management / multi-tenant key vault.
- Anthropic-backed transcription (Whisper path is Upgrade 04).
- Swapping the default backend to cloud. Never.

---

## 8. First action for the session

State the plan in one message: which files change, in what order, and the gate test you will write first. Then start at §5 step 1. Do not touch the Anthropic SDK until the schema floor and `OllamaProvider` wrapper are green.
