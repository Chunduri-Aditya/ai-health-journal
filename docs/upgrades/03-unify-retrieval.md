# Upgrade 03 — Unify the Retrieval Architecture

## Problem

The repo has **two parallel retrieval systems**:

- **Legacy** `rag_store.py` — what `app.py` actually uses via `get_rag_store()`. String-joined retrieval result, no namespace support, concatenated entry+insight blob.
- **Modern** `vector_store/` — factory pattern (`none`/`chroma`/`pinecone`), structured `List[Dict]` result shape, namespace-aware, used by **nothing in the app today**.

```10:20:vector_store/factory.py
def get_vector_store(namespace: Optional[str] = None) -> VectorStore:
    """
    Return a concrete VectorStore implementation based on VECTOR_BACKEND.

    VECTOR_BACKEND=chroma   -> local Chroma (requires chromadb)
    VECTOR_BACKEND=pinecone -> Pinecone + LangChain integration (requires optional deps)
    VECTOR_BACKEND=none     -> returns a no-op store (retrieval disabled)

    The optional `namespace` is forwarded to the backend when supported
    (used for per-user or per-session isolation).
    """
```

Every future retrieval-touching change is now a choice between (a) evolving the dead module, (b) evolving the used module, or (c) doing both. This is how drift becomes bugs.

## Goal

Exactly one retrieval surface in the running app. `app.py` consumes `vector_store/` only. `rag_store.py` is deleted. The interface is structured, namespace-aware, and type-consistent across `none` / `chroma` / `pinecone`.

## Dependencies

- **Must land before:** `01-retrieval-grounding.md` (so the grounding fix lands on the surviving surface).
- **Benefits from:** `06-tests-schema.md` subset (contract tests against a fake `VectorStore` to pin behavior before surgery).

## Plan

### A. Finalize the `VectorStore` interface

**`vector_store/base.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class RetrievalHit:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class VectorStore(ABC):
    @abstractmethod
    def add_entry(
        self,
        entry_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        namespace: Optional[str] = None,
    ) -> None: ...

    @abstractmethod
    def query(
        self,
        text: str,
        *,
        top_k: int = 3,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalHit]: ...

    @abstractmethod
    def delete_entry(self, entry_id: str, *, namespace: Optional[str] = None) -> None: ...

    @abstractmethod
    def clear_namespace(self, namespace: str) -> None: ...

    @abstractmethod
    def healthcheck(self) -> bool: ...

    @property
    @abstractmethod
    def enabled(self) -> bool: ...
```

Design choices:
- Return **`RetrievalHit` dataclass**, not bare dict. Backends can still build a dict internally and hand it off.
- `enabled` is a property so the noop backend is honest about its state without the caller special-casing.
- `healthcheck` returns `bool`, does not raise. Used for `/ping` expansion.
- `namespace` is an explicit kwarg everywhere. No more "current namespace" singleton state.
- `score` ordering is meaningful within one backend + query only. Treat it as a relative ranking signal, not a normalized confidence value portable across Chroma and Pinecone.

### B. Implementations

**`vector_store/noop_store.py`** (new — promote the anonymous class out of the factory):

```python
class NoOpStore(VectorStore):
    @property
    def enabled(self) -> bool:
        return False
    def add_entry(self, *a, **kw): return None
    def query(self, *a, **kw) -> List[RetrievalHit]: return []
    def delete_entry(self, *a, **kw): return None
    def clear_namespace(self, namespace: str): return None
    def healthcheck(self) -> bool: return True
```

**`vector_store/chroma_store.py`**

- One `PersistentClient` at `./storage/chroma/` (align with 02's `./storage/` home).
- `namespace` → one collection per namespace (`get_or_create_collection(name=f"ns__{ns}")`). Rationale: Chroma metadata filters (`where={"namespace": ns}`) force a full-collection scan for every query; per-collection isolation scales better, is trivially deleteable via `delete_collection`, and matches Pinecone's per-namespace mental model.
- `filter_metadata` → passed through as `where=...`.
- `add_entry` uses the caller-supplied `entry_id`.

**`vector_store/pinecone_store.py`**

- `namespace` → native Pinecone `namespace=` parameter.
- `filter_metadata` → Pinecone metadata filter syntax.

### C. App integration (`app.py`)

Replace:

```python
from rag_store import get_rag_store
rag_store = get_rag_store(enabled=cfg.retrieval_enabled)
```

with:

```python
from vector_store.factory import get_vector_store
vector_store: VectorStore = get_vector_store()
```

Factory honors `cfg.vector_backend` and `cfg.retrieval_enabled` together: if `retrieval_enabled=False` OR `vector_backend=none`, return `NoOpStore`. Callers can then always call `vector_store.add_entry(...)` / `.query(...)` without guards — the noop store silently drops writes and returns `[]`.

### D. Factory consolidation (`vector_store/factory.py`)

```python
def get_vector_store() -> VectorStore:
    cfg = load_config()
    if not cfg.retrieval_enabled or cfg.vector_backend == "none":
        return NoOpStore()
    if cfg.vector_backend == "chroma":
        try:
            from .chroma_store import ChromaStore
        except ImportError as e:
            logging.error("Chroma selected but unavailable; falling back to noop. %s", e)
            return NoOpStore()
        return ChromaStore()
    if cfg.vector_backend == "pinecone":
        if not cfg.allow_cloud_vectorstore:
            raise RuntimeError("cloud_vectorstore_not_enabled: set ALLOW_CLOUD_VECTORSTORE=true to enable Pinecone.")
        try:
            from .pinecone_store import PineconeStore
        except ImportError as e:
            raise RuntimeError(f"Pinecone backend unavailable: {e}") from e
        return PineconeStore()
    logging.warning("Unknown VECTOR_BACKEND=%r; using noop.", cfg.vector_backend)
    return NoOpStore()
```

Note: single module-level singleton not required — both Chroma and Pinecone clients cache their own connections.

### E. Deletions

- Delete `rag_store.py` after the migration.
- Keep `tools/pinecone_bootstrap.py` (independent utility).

### F. Evals integration

`evals/run_evals.py` imports `from rag_store import get_rag_store, CHROMA_AVAILABLE`. Update to:

```python
from vector_store.factory import get_vector_store
vs = get_vector_store()
rag_enabled = vs.enabled
retrieved_context = vs.query(entry, top_k=3, namespace="eval")
```

Format hits into the `retrieved_context` string the evaluator already expects, preserving all metric semantics.

## New / changed interfaces

### `VectorStore` (canonical, as shown above).

Note: the current checked-in `vector_store/base.py` still returns `List[Dict[str, Any]]`. This track deliberately upgrades that surface to `RetrievalHit` so downstream code can rely on typed access.

### `/ping` upgrade (optional but cheap)

```json
GET /ping
{
  "status": "ok",
  "version": "...",
  "retrieval": { "backend": "chroma", "healthy": true, "enabled": true }
}
```

## Acceptance criteria

1. `grep -rn "from rag_store"` returns zero hits in the app (tests permitted to keep mocks referencing its old shape only transiently).
2. `rag_store.py` is deleted.
3. Switching `VECTOR_BACKEND=none` / `chroma` requires no code changes in `app.py` or any route.
4. A contract test (from 06) runs `add_entry → query → delete_entry → clear_namespace` against a fake `VectorStore` and against `ChromaStore` (marked `@pytest.mark.integration`, opt-in).
5. Two namespaces (`session:A` and `session:B`) do not return each other's documents.
6. `/ping` reports `retrieval.backend` and `retrieval.healthy`.
7. Evals (`make eval-smoke` or its replacement) still pass against the unified surface.

## Risks & open questions

- **Chroma persistence directory.** Today `rag_store.py` uses `./rag_store/`; `vector_store/chroma_store.py` should use `./storage/chroma/` to align with 02. Migration is not required (no production data assumed). Document the path change.
- **LangChain-Pinecone version pin.** `requirements-optional.txt` pins `langchain-pinecone==0.2.13`. Keep the pin; any bump is a separate change.
- **Embedding function.** Chroma's default is `all-MiniLM-L6-v2`; Pinecone path uses whatever `langchain-pinecone` + `sentence-transformers` provides. Document in 07's "behavior parity check."
- **Namespace cardinality.** If `RAG_NAMESPACE_MODE=session` in high-traffic usage, per-session collections could explode. Not a concern for local single-user use. Flag for future.

## Touch list

- `vector_store/base.py` — expand interface, add `RetrievalHit`.
- `vector_store/factory.py` — consolidate logic; drop inline `NoOpStore`.
- `vector_store/noop_store.py` — new.
- `vector_store/chroma_store.py` — implement per-namespace collections, `delete_entry`, `clear_namespace`, `healthcheck`.
- `vector_store/pinecone_store.py` — implement `delete_entry`, `clear_namespace`, `healthcheck`.
- `app.py` — swap import, drop guards, update `_retrieve_context` / `_store_in_rag` (01 completes that).
- `evals/run_evals.py` — swap import, update retrieval block.
- `rag_store.py` — **delete**.
- `tests/vector_store/` — contract + integration tests (under 06).
- `README.md` — update `## RAG store` / `Vector backends` section; note path change.
