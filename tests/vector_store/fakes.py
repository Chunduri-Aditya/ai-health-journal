"""
Deterministic, in-memory VectorStore for tests.

Implements the `vector_store.base.VectorStore` interface introduced in
upgrade 03: namespace + filter_metadata + delete_entry + clear_namespace +
healthcheck + enabled/backend_name properties, and returns `RetrievalHit`.

Scoring is Jaccard overlap on lowercased tokens so rankings are
predictable without an embedding model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from vector_store.base import RetrievalHit, VectorStore


def _tokens(text: str) -> set:
    return {t for t in text.lower().split() if t}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union) if union else 0.0


def _matches_filter(meta: Dict[str, Any], flt: Optional[Dict[str, Any]]) -> bool:
    if not flt:
        return True
    return all(meta.get(k) == v for k, v in flt.items())


class InMemoryVectorStore(VectorStore):
    """Minimal VectorStore useful for deterministic unit tests."""

    def __init__(self, default_namespace: str = "default") -> None:
        # namespace -> { entry_id: (text, metadata) }
        self._docs: Dict[str, Dict[str, Tuple[str, Dict[str, Any]]]] = {}
        self._default_namespace = default_namespace
        self._healthy = True

    # ── VectorStore properties ────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return True

    @property
    def backend_name(self) -> str:
        return "in_memory"

    def _ns(self, namespace: Optional[str]) -> str:
        return namespace or self._default_namespace

    # ── VectorStore methods ───────────────────────────────────────────────────
    def add_entry(
        self,
        entry_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        namespace: Optional[str] = None,
    ) -> bool:
        ns = self._ns(namespace)
        bucket = self._docs.setdefault(ns, {})
        bucket[entry_id] = (text, dict(metadata or {}))
        return True

    def query(
        self,
        text: str,
        *,
        top_k: int = 3,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalHit]:
        if top_k <= 0:
            return []
        ns = self._ns(namespace)
        bucket = self._docs.get(ns, {})
        scored = [
            RetrievalHit(
                id=doc_id,
                text=doc_text,
                score=_jaccard(text, doc_text),
                metadata=dict(meta),
            )
            for doc_id, (doc_text, meta) in bucket.items()
            if _matches_filter(meta, filter_metadata)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def delete_entry(
        self,
        entry_id: str,
        *,
        namespace: Optional[str] = None,
    ) -> None:
        ns = self._ns(namespace)
        bucket = self._docs.get(ns)
        if bucket is not None:
            bucket.pop(entry_id, None)

    def clear_namespace(self, namespace: str) -> None:
        self._docs.pop(namespace or self._default_namespace, None)

    def healthcheck(self) -> bool:
        return self._healthy

    # ── Test helpers (not part of the interface) ─────────────────────────────
    def set_healthy(self, value: bool) -> None:
        self._healthy = value

    def all_ids(self, namespace: Optional[str] = None) -> List[str]:
        return list(self._docs.get(self._ns(namespace), {}).keys())
