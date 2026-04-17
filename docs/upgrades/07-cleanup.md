# Upgrade 07 — Clean Up Dead or Drifted Paths

## Problem

Several modules, flags, and README passages describe behavior that doesn't match runtime. Each one individually is minor. Together they make the project hostile to new contributors and unsafe to refactor.

Known drift inventory:

| #  | Defect                                                                                                | Location                                     | Disposition |
|----|-------------------------------------------------------------------------------------------------------|----------------------------------------------|-------------|
| 1  | `chains/insight_chain.py` fallback imports `DRAFT_SYSTEM_PROMPT` from `llm_client` (wrong module)     | `chains/insight_chain.py:104`                | Fix or delete |
| 2  | `Makefile::verify` uses `--mock_llm --verify-only` flags not implemented in `run_evals.py`            | `Makefile:20`                                | Implement or remove |
| 3  | `Makefile::eval-smoke` uses `--mode langchain_chain` flag not implemented in `run_evals.py`           | `Makefile:24–26`                             | Implement or remove |
| 4  | `Makefile::eval-smoke-retrieval` same as above                                                        | `Makefile:30–32`                             | Implement or remove |
| 5  | `Makefile::report` reads from `artifacts/test_runs/evals/` — the tree writes to `evals/results/`      | `Makefile:36–43`                             | Align paths  |
| 6  | README example `.env` says `RETRIEVAL_ENABLED=true`; code defaults to `false`                         | `README.md`, `config.py:39`                  | Align docs or flip default |
| 7  | `privacy/local_text_cache.py` is only consumed by the (gated) Pinecone path in `vector_store/pinecone_store.py`; deletable only after that consumer is removed or rewritten | `privacy/local_text_cache.py`, `vector_store/pinecone_store.py:12,112` | Keep until 03 revisits pinecone_store |
| 8  | `privacy/redact.py` has no callers in the running app                                                 | `privacy/redact.py`                          | Wire in or delete |
| 9  | `behavior/` rules/patterns/few-shots exist but aren't loaded by `/analyze`                            | `behavior/*`                                 | Wire in or scope as future |
| 10 | `README.md` describes legacy/baseline/quality mode but doesn't describe the `baseline_json_mode` flag clearly | `README.md` + `/analyze` table          | Update docs |
| 11 | `requirements.txt` exists alongside `requirements-core.txt` / `requirements-optional.txt` as "legacy" | root                                         | Delete or clarify |
| 12 | `AnalysisOutput` pydantic defined but not used in live path (overlaps with 06)                        | `schemas/analysis.py`                        | Handled in 06; verify after merge |
| 13 | `rag_store.py` (handled in 03) still referenced by `evals/run_evals.py` (`from rag_store import ...`) | `evals/run_evals.py`                         | Handled in 03; verify |
| 14 | README "Clean Folder" section describes a dead `clean_export/` workflow of questionable value         | `README.md`                                  | Prune or move to CONTRIBUTING.md |

## Goal

Every file that ships with the repo either:
- Runs on a common code path, or
- Is covered by a test (under 06), or
- Is explicitly deleted.

Every documented claim in `README.md` matches actual runtime behavior.

## Dependencies

- **Rolling.** Each of 01–06 resolves its slice of 07 as part of its acceptance criteria. A final sweep runs after the others merge.
- **Benefits from:** 06 tests (safer deletion).

## Plan

### A. `chains/insight_chain.py`

**Decision required:** is LangChain a supported code path or not?

- If **yes** — wire `run_insight_chain` into `/analyze` as an alternate mode (e.g. `chain_mode: true`), update the fallback import to `from generator_prompts import DRAFT_SYSTEM_PROMPT` (and keep `json_generate` from `llm_client`), add at least one test that exercises the fallback.
- If **no** — delete `chains/insight_chain.py`, drop the LangChain imports from `requirements-*.txt` if no other module needs them, drop README references.

Recommended: **delete**. The main pipeline already does everything this chain does, and the fallback path is broken today — strong evidence no one uses it.

### B. Makefile targets referencing unimplemented flags

Two options per flag:

- `--mock_llm` — valuable for fast CI. Implement it in `evals/run_evals.py`: if set, patch `llm_client.ollama_generate` to return canned fixtures keyed on model + mode. Reuse the same fixtures from `tests/routes/test_analyze_modes.py`.
- `--verify-only` — weak signal; probably meant "run but don't write output files". Replace with a `--dry-run` flag or delete.
- `--mode langchain_chain` — contingent on section A's decision.

Recommendation:
- Implement `--mock_llm` and `--dry-run`.
- Delete `langchain_chain` mode and the two Makefile targets that reference it; replace `eval-smoke` with a `--mock_llm` variant that exercises `baseline_json` + `quality`.

### C. Makefile `report` target path mismatch

`make report` reads `artifacts/test_runs/evals/*.json`; `evals/run_evals.py` writes `evals/results/*.json`. Either:

- Change the report target to read from `evals/results/`, or
- Change `run_evals.py` to write to `artifacts/test_runs/evals/` (aligns better with general "build artifacts go in `artifacts/`" conventions).

Recommendation: the second. Update `.gitignore` to include `artifacts/`.

### D. README alignment sweep

- Set the `.env` example's `RETRIEVAL_ENABLED` to match the default (`false`), with a short explainer of what flipping to `true` enables and requires.
- Add a `baseline_json_mode` row to the API-endpoints table.
- Update the "Architecture" mermaid diagram to show SQLite (post-02), SSE events (post-05), and the unified `vector_store/` (post-03).
- Remove or shorten the "Clean Folder" rsync section; if kept, move to `CONTRIBUTING.md`.
- Remove the stale `openai-whisper` guidance (post-04).
- Remove the dual `requirements.txt` / `requirements-core.txt` confusion by deleting `requirements.txt` and adding a single note in README.

### E. `privacy/` module disposition

Two files. Current call-graph:

- **`privacy/redact.py`** — no callers in the running app. Candidate for wiring in.
- **`privacy/local_text_cache.py`** — **not dead**. Imported and called by `vector_store/pinecone_store.py:12,112`. Writes and reads a JSONL cache of original texts so Pinecone retrieval can round-trip the source string even when only embeddings live remote. Removing it would break the Pinecone path.

Decisions:

- **Wire in:** `privacy/redact.py` runs before every `vector_store.add_entry` (post-03), and optionally on export/share paths. Keep SQLite as the canonical local journal record unless the user explicitly asks for full-storage redaction. Config: `PRIVACY_REDACT=off|vector_only|full`.
- **Keep `privacy/local_text_cache.py`** until upgrade 03's retrieval-unification work revisits `pinecone_store.py`. At that point, evaluate whether the cache still has a purpose (post-03 the SQLite `entries.body` column may cover the same use case).

Recommendation: wire in `redact.py` with a conservative default (`off` for now, with `vector_only` as the first supported mode once hardened). Leave `local_text_cache.py` alone for this pass. Add a test under 06.

### F. `behavior/` module disposition

`behavior/rules.json`, `failure_patterns.json`, `few_shot.jsonl`, and `loader.py` exist but aren't invoked by `/analyze`. They're meant to be populated by `tools/distill_evals_to_behavior.py` → consumed as additional grounding in prompts.

Decision: **keep, but scope as a future upgrade (08?).** Don't delete — the `distill` tool produces real value for DPO iteration. But add a `README.md` inside `behavior/` that says so explicitly, so the next reader doesn't mistake it for dead code.

### G. Compatibility with removed symbols

Any deletion must be grep-checked:

```bash
git grep -n "from rag_store"              # should be zero after 03
git grep -n "import whisper"              # should be zero after 04
git grep -n "DRAFT_SYSTEM_PROMPT"         # should only appear in generator_prompts.py and imports of it
git grep -n "mock_llm\|verify-only\|langchain_chain"
git grep -n "local_text_cache"
git grep -n "openai-whisper\|pip install openai-whisper"
```

Add these to the `make verify` checklist (a `scripts/check_drift.sh` that grep-exits non-zero on hits).

## New / changed interfaces

- `evals/run_evals.py` gains `--mock_llm`, `--dry-run`.
- `privacy/redact.py` gains a single public `redact(text: str, mode: str) -> str` used by vector-store writes and export/share paths, with full-storage redaction as an explicit opt-in only.
- `privacy/README.md` — new, documents the boundary between canonical local storage and redacted retrieval copies.
- `scripts/check_drift.sh` — new, invoked by `make verify`.

## Acceptance criteria

1. `git grep -n "from rag_store"` returns zero (confirms 03).
2. `git grep -n "openai-whisper"` returns zero (confirms 04).
3. `make verify` on a clean tree runs compile + drift check + pytest, and exits 0.
4. Every file under `chains/`, `privacy/`, `behavior/` is either imported from a runtime path or has a sibling `README.md` marking it as a deliberate future-work staging area.
5. README's `.env` example matches `config.py` defaults (or README explicitly says "if you want retrieval, flip these two toggles").
6. No Makefile target references a flag that `run_evals.py` doesn't implement.
7. `requirements.txt` either matches `requirements-core.txt` exactly, or is deleted with a README note.

## Risks & open questions

- **Deleting LangChain.** If any future plan involves multi-agent LangGraph orchestration, section A's "delete" path costs rework. Confirm with stakeholder before removal.
- **Behavior distillation.** If `tools/distill_evals_to_behavior.py` is run periodically, its outputs need a place. The `behavior/` folder should stay. Adding a `behavior/README.md` with a "how this is regenerated" note is the minimum non-negotiable.
- **Redaction defaults.** Turning redaction on by default changes the DPO dataset substance (redacted text ≠ original text). Keep default `off`, document, and make it a deliberate toggle.
- **Canonical record semantics.** Storing only redacted text in SQLite would degrade the core journaling product. Keep the default boundary explicit: original text in local DB, optionally redacted text in retrieval/indexing layers.

## Touch list

- `chains/insight_chain.py` — **archived** to `archive/chains/insight_chain.py` on 2026-04-17 (see `archive/README.md`).
- `privacy/local_text_cache.py` — keep; reassess during upgrade 03 after `pinecone_store.py` is touched.
- `privacy/redact.py` — keep; wire into `vector_store` write path and export/share flow, with any `storage/repository.py` use gated behind explicit `full` mode.
- `privacy/README.md` — new.
- `behavior/README.md` — new.
- `Makefile` — drop `langchain_chain`-related targets; align `report` path; rewrite `verify`; rewrite `eval-smoke` around `--mock_llm`.
- `evals/run_evals.py` — implement `--mock_llm`, `--dry-run`; drop `langchain_chain` mode.
- `README.md` — alignment sweep per section D.
- `requirements.txt` — **archived** to `archive/requirements.txt` on 2026-04-17 (README note still pending).
- `scripts/check_drift.sh` — new.
- `.gitignore` — add `artifacts/`, `storage/aihj.sqlite3*`, `storage/secret.key`, `storage/chroma/`.
