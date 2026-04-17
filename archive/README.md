# Archived Paths

This directory holds files retired from the live code tree. They remain in
the repository (and in `git log --follow`) for history and for the occasional
"what did we used to do here?" inspection. Nothing in `archive/` is imported,
executed, or tested.

If you are looking for the current design rationale, read
[`../docs/upgrades/07-cleanup.md`](../docs/upgrades/07-cleanup.md).

---

## Inventory

| Path                                   | Retired on     | Replaced by                                                | Doc reference                                    |
|----------------------------------------|----------------|------------------------------------------------------------|--------------------------------------------------|
| `archive/chains/insight_chain.py`      | 2026-04-17     | Main `/analyze` pipeline in `app.py` (same functionality, no broken fallback import). | [`07-cleanup.md § A`](../docs/upgrades/07-cleanup.md) |
| `archive/requirements.txt`             | 2026-04-17     | `requirements-core.txt` + `requirements-optional.txt` + `requirements-dev.txt`. | [`07-cleanup.md § D`](../docs/upgrades/07-cleanup.md) |

### Why `chains/insight_chain.py` was archived

- It defined a LangChain-based alternate path for `/analyze` that was never wired
  into any route.
- The non-LangChain fallback imported `DRAFT_SYSTEM_PROMPT` from `llm_client`,
  but that symbol lives in `generator_prompts.py`. The fallback path raised
  `ImportError` on first use — strong evidence nothing used it.
- The main pipeline (`_run_quality_pipeline` in `app.py`) already provides the
  functionality the chain aimed at.

### Why the root `requirements.txt` was archived

- The README already labeled it "legacy".
- Content was a partial superset of `requirements-core.txt`, with `chromadb`
  pinned in, which muddied the core/optional split.
- `train/requirements.txt` is **still live** and remains where it is — it is
  the training stack, not part of the app runtime.

---

## Candidates that were *not* archived

These looked dead at first inspection but turn out to have live consumers.
Documented here so the next person doesn't make the same mistake:

- **`privacy/local_text_cache.py`** — imported and called by
  `vector_store/pinecone_store.py` (L12, L112). Lives under the Pinecone
  feature flag but is otherwise active. Revisit after upgrade 03 unifies
  the retrieval backends.
- **`rag_store.py`** — still the active retrieval backend used by `app.py`
  and `evals/run_evals.py`. Will be retired by upgrade 03.
- **`privacy/redact.py`** — upgrade 07 recommends wiring this into the
  retrieval write path rather than deleting it.
- **`behavior/*`** — consumed (indirectly) by `tools/distill_evals_to_behavior.py`.
  Kept as a staging area for the behavior-distillation pipeline.

---

## Restoring a file from `archive/`

```bash
git mv archive/chains/insight_chain.py chains/insight_chain.py
```

Then re-wire it into `app.py` imports. The fallback import bug noted above
must be fixed if restoration is attempted.
