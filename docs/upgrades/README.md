# Upgrade Roadmap — Index

Seven tracks covering product quality, reliability, and maintainability for the AI Health Journal repo.

Each `NN-*.md` file is **self-contained**: a fresh agent session (or new contributor) can read one file and execute the track without the rest of this chat history. Cross-track dependencies are called out explicitly in each doc's "Dependencies" section.

Before reading any track, read [`../PROJECT_OVERVIEW.md`](../PROJECT_OVERVIEW.md) for shared context.

This directory is implementation-facing. It mixes current-state audit details with proposed target-state designs; unless a snippet is explicitly labeled as current behavior, treat inline code blocks as the intended end state for that track.

Line references and defect inventory were verified against tree snapshot `a17105b` on 2026-04-17.

---

## Tracks

| #  | Title                                                      | File                                   | Impact class      |
|----|------------------------------------------------------------|----------------------------------------|-------------------|
| 01 | Fix retrieval grounding (use current entry as query)       | [`01-retrieval-grounding.md`](01-retrieval-grounding.md) | Product quality   |
| 02 | Replace cookie session history with SQLite persistence     | [`02-sqlite-persistence.md`](02-sqlite-persistence.md)   | Reliability       |
| 03 | Unify the retrieval architecture (`vector_store/` only)    | [`03-unify-retrieval.md`](03-unify-retrieval.md)         | Maintainability   |
| 04 | Repair and modernize voice transcription                   | [`04-transcription.md`](04-transcription.md)             | Reliability       |
| 05 | Replace fake pipeline progress with real SSE streaming     | [`05-sse-streaming.md`](05-sse-streaming.md)             | Product quality   |
| 06 | Real test coverage + runtime schema validation             | [`06-tests-schema.md`](06-tests-schema.md)               | Reliability       |
| 07 | Clean up dead / drifted paths                              | [`07-cleanup.md`](07-cleanup.md)                         | Maintainability   |

---

## Recommended delivery order

> **06a (subset inside 06) → 03 → 01 → 02 → 05 → 04 → 07 (rolling)**

Rationale:

1. **06a (subset first)** — Before any refactor, land a minimal pytest floor + post-parse validation of `AnalysisOutput` / `VerifierVerdict`. This is the safety net that lets 03 and 01 land without silently breaking DPO pair generation.
2. **03 unify retrieval** — Delete `rag_store.py` and route `app.py` through `vector_store/`. Do this *before* 01 so the grounding fix lands on the surface that will still exist next month.
3. **01 fix grounding** — On the unified retrieval surface: use the current entry as the query, exclude the just-submitted entry from its own context, return structured sources. This is the single highest-impact product change.
4. **02 SQLite persistence** — Replace cookie-backed session history with a local SQLite store. Natural follow-on to 01 because the new `retrieval_hits` table lets the UI audit which sources informed each analysis.
5. **05 SSE streaming** — With schema validation in place (from 06), typed pipeline events become trivial. Also fixes the broken `stageTimer` double-assignment bug in the current UI.
6. **04 transcription** — Isolated; can be done in parallel with any of 02/05 once a developer is free.
7. **07 cleanup** — Rolling. Every track that touches a drifted area resolves its slice of 07 as part of its acceptance criteria. A final sweep at the end handles leftovers.

### Why not the original order?

The original proposed order put retrieval refactors (1, 3) first and tests (6) fourth — i.e. two large refactors ship untested, then tests arrive after the risk window has closed. Inverting pays ~1 day of test scaffolding up front to de-risk the next five weeks.

---

## Shared conventions

All upgrade docs follow the same structure:

1. **Problem** — what's broken today, with exact file + line references.
2. **Goal** — one-paragraph description of the target state.
3. **Dependencies** — other upgrades that must land first, or are safe to parallelize.
4. **Plan** — step-by-step changes (file-by-file where useful).
5. **New / changed interfaces** — signatures, schemas, API shapes.
6. **Acceptance criteria** — checkable conditions.
7. **Risks & open questions** — known unknowns, migration concerns.
8. **Touch list** — every file expected to change.

Additional conventions:

- The docs distinguish **today's code** from **proposed implementation**. Sample code in a track should be assumed to describe the target implementation for that track unless the section says otherwise.
- Acceptance criteria should be verifiable on a developer machine without relying on conversational context from this thread.
- If a track depends on another track's abstractions, prefer updating the dependent doc rather than repeating obsolete names from the current codebase.

---

## Status tracker

Update this table as tracks complete:

| #  | Status     | PR / branch | Notes                                              |
|----|------------|-------------|----------------------------------------------------|
| 01 | not started |             |                                                    |
| 02 | not started |             |                                                    |
| 03 | not started |             |                                                    |
| 04 | not started |             |                                                    |
| 05 | not started |             |                                                    |
| 06 | not started |             |                                                    |
| 07 | not started |             | Rolling; tracked as sub-items on other PRs         |
