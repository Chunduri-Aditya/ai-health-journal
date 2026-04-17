# Upgrade 01 — Fix Retrieval Grounding

## Problem

`app.py::_retrieve_context()` at L345–348 calls:

```345:348:app.py
def _retrieve_context(top_k: int = 3) -> str:
    if cfg.retrieval_enabled and rag_store.enabled:
        return rag_store.retrieve("", top_k=top_k)
    return ""
```

The empty query string means Chroma returns effectively arbitrary prior documents. The "quality" pipeline then verifies/revises the draft against ungrounded memory, which undermines the project's central claim ("grounded in your past entries, reduced hallucinations").

Related symptoms:

- `rag_store.py::add_entry` (L58) stores a single concatenated `"ENTRY: {entry}\n\nINSIGHT: {insight}"` blob with timestamp metadata only — no stable record id, no session/user namespace, no source for UI display.
- `rag_store.py::retrieve` (L92) returns a single joined string; the caller cannot audit which sources were used.
- The just-submitted entry gets added to the store *after* retrieval today, but there's no source filter protecting against self-retrieval in future configurations (e.g. if write-then-query is ever reordered for streaming).

## Goal

`/analyze` retrieves context using the **current journal entry** as the query, **excludes the just-submitted entry**, returns **structured source objects** (not a joined blob), and **namespace-isolates** by session/user. The response body includes the sources so the UI can render them.

## Dependencies

- **Must land after:** `03-unify-retrieval.md`. Grounding fix lands on the unified `vector_store/` interface so the work isn't rewritten when `rag_store.py` is retired.
- **Benefits from:** `06-tests-schema.md` subset (contract tests for retrieval behavior against a fake backend).
- **Enables:** `02-sqlite-persistence.md` can persist a `retrieval_hits` table referencing the structured source ids emitted here.

## Plan

### A. Change `_retrieve_context` signature and call sites

**`app.py`**

```python
from vector_store.base import RetrievalHit

def _retrieve_context(
    journal_entry: str,
    session_id: str,
    *,
    top_k: int = 3,
    exclude_ids: Optional[Set[str]] = None,
) -> List[RetrievalHit]:
    if not vector_store.enabled:
        return []
    hits = vector_store.query(
        text=journal_entry,
        top_k=top_k,
        namespace=_namespace_for(session_id),
    )
    if exclude_ids:
        hits = [h for h in hits if h.id not in exclude_ids]
    return hits[:top_k]
```

Call sites to update:

- `_run_quality_pipeline` (L219) — pass `journal_entry`, session id, and exclude any id equal to the pending entry (not usually present at read time, but future-proofs streaming reordering).
- `_run_baseline` (L288) — same.
- `_run_legacy` (L323) — same.

### B. Prompt integration

Keep the existing prompt API accepting a string, but add a helper to format structured sources:

**`generator_prompts.py`**

```python
def format_sources_for_prompt(sources: List[RetrievalHit]) -> str:
    if not sources:
        return ""
    lines = []
    for i, s in enumerate(sources, 1):
        ts = s.metadata.get("created_at", "")
        lines.append(f"[Source {i} | {ts}]\n{s.text}\n")
    return "\n".join(lines)
```

`get_draft_prompt` and `get_verifier_prompt` accept `retrieved_context: str` as today; the caller runs `format_sources_for_prompt(...)` first. This keeps prompt surface stable (DPO-stable) and isolates the shape change to the app layer.

### C. Storage schema change

The unified `VectorStore.add_entry` already takes `(entry_id, text, metadata)`. Standardize metadata for journal entries:

```python
metadata = {
    "kind": "journal_entry",
    "session_id": session_id,
    "user_id": user_id or "local",
    "created_at": datetime.utcnow().isoformat(),
    "generator_model": generator_model,
    "entry_length": len(entry),
    "analysis_id": analysis_id,  # optional back-pointer (filled once 02 lands)
}
vector_store.add_entry(entry_id=new_id, text=entry, metadata=metadata)
```

The stored text is the **entry only**, not `"ENTRY: ... INSIGHT: ..."`. Rationale: we're retrieving *past entries* to ground the current one; the insight is derivative and should not dominate similarity scoring. Insights are persisted in SQLite (see `02-sqlite-persistence.md`), linked by `analysis_id`.

### D. Self-exclusion

Option 1 (preferred): **write-after-retrieval** ordering.
Option 2: pre-compute `new_id` and pass `exclude_ids={new_id}` into `_retrieve_context`, then write. Needed for the streaming track (05) where retrieval happens after write for cache-warming reasons.

Pick option 2 because it's robust to future reordering.

Important: self-exclusion is **id-based**, not text-based. A prior historical entry with identical text is still a legitimate retrieval candidate if it has a different `entry_id`.

### E. Namespace isolation

`_namespace_for(session_id)` switches on `cfg.rag_namespace_mode`:

| mode       | namespace                                        |
|------------|--------------------------------------------------|
| `session`  | `f"session:{session_id}"`                        |
| `user`     | `f"user:{request.headers.get(cfg.rag_user_id_header, 'anonymous')}"` |
| `fixed`    | `cfg.rag_namespace_fixed`                        |

Passed to `vector_store.query(..., namespace=...)`. Chroma backend translates this to a `where={"namespace": ns}` filter or a per-namespace collection (decide in 03).

### F. API change

`/analyze` response gains a `sources` field:

```json
{
  "insight": "…",
  "analysis": { … },
  "sources": [
    {
      "id": "entry_2026-04-17T08:12:33",
      "score": 0.81,
      "snippet": "First 240 chars of the source entry…",
      "created_at": "2026-04-17T08:12:33",
      "generator_model": "phi3:3.8b"
    }
  ]
}
```

UI work is out of scope for this track (belongs to 05 UI pass or a dedicated UI ticket), but the API must emit the field.

## New / changed interfaces

### `_retrieve_context`

```python
def _retrieve_context(
    journal_entry: str,
    session_id: str,
    *,
    top_k: int = 3,
    exclude_ids: Optional[Set[str]] = None,
) -> List[RetrievalHit]: ...
```

### `VectorStore.query` (from upgrade 03)

```python
def query(
    self,
    text: str,
    *,
    top_k: int = 3,
    namespace: Optional[str] = None,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[RetrievalHit]:
    """
    Returns retrieval hits such as:
      RetrievalHit(id=str, text=str, score=float, metadata={...})
    sorted descending by score.
    """
```

### `/analyze` response schema delta

- Adds `sources: List[Dict[str, Any]]` (possibly empty if RAG disabled).
- `snippet` is derived server-side from `RetrievalHit.text`; it is not a second source of truth stored separately in the vector backend.

## Acceptance criteria

1. `_retrieve_context` is called with the current entry text; grep for `retrieve("",` returns zero hits.
2. A same-request self-hit is excluded by `entry_id` even in a write-before-query simulation. Historical duplicates with different ids may still appear.
3. `/analyze` response includes a `sources` array of structured objects matching the schema above.
4. On an eval run with RAG enabled against `evals/hard_negatives_hn_v2.jsonl`:
   - `context_precision` mean increases vs pre-fix baseline (≥ 0.20 absolute improvement on RAG-dependent cases, or document why not).
   - `faithfulness` mean ≥ 0.95 in quality mode.
5. With `RAG_NAMESPACE_MODE=session` and two simulated sessions, neither session's entries appear in the other session's retrieval results.
6. Contract test (from 06) verifies the `exclude_ids` filter removes requested ids from results.

## Risks & open questions

- **Chroma namespace strategy.** `chromadb` supports metadata filters (`where={"namespace": ns}`) per-collection or per-`client.get_or_create_collection(name=ns)`. Per-collection scales better for many namespaces; metadata filter is simpler. Decide in 03 and document.
- **Embedding choice.** `chromadb` defaults to `all-MiniLM-L6-v2`. No change unless eval shows retrieval quality is limiting `faithfulness`.
- **Cold start.** A brand-new user has no prior entries → retrieval returns empty → current prompt template handles this (`"(no prior context retrieved)"`). Verify both `get_draft_prompt` and `get_verifier_prompt` handle empty context gracefully.
- **PII leakage via retrieval.** If `privacy/redact.py` is ever wired in (07), it must run before `add_entry` writes text to the vector store. Flag for 02/07.
- **Snippet policy.** The API should expose a preview, not necessarily the full source text, when rendering audit UI. Decide a stable truncation rule (for example 240 chars + ellipsis) and keep it out of the embedding payload.

## Touch list

- `app.py` — `_retrieve_context`, `_store_in_rag`, `_run_quality_pipeline`, `_run_baseline`, `_run_legacy`, `/analyze` response.
- `rag_store.py` — **deleted in 03**; don't invest further changes here.
- `vector_store/base.py` — add `namespace` and `filter_metadata` kwargs (landed in 03).
- `vector_store/chroma_store.py` — implement namespace + metadata filter (landed in 03).
- `vector_store/pinecone_store.py` — implement namespace (native Pinecone concept).
- `generator_prompts.py` — add `format_sources_for_prompt`.
- `evals/run_evals.py` — switch retrieval call to new interface; nothing else should change.
- `templates/index.html` — add a minimal "Sources used" section (optional; can defer).
