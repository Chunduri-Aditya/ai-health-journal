"""
Deterministic, in-memory VectorStore for tests.

Implements the current `vector_store.base.VectorStore` interface
(add_entry, query). Upgrade 03 will widen the interface with
namespace, filter_metadata, delete_entry, clear_namespace, and
healthcheck; when it lands, this fake should grow alongside the base
class so contract tests stay meaningful.

Scoring is a simple token-overlap (Jaccard on lowercased tokens) so
rankings are predictable without an embedding model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from vector_store.base import VectorStore


def _tokens(text: str) -> set:
    return {t for t in text.lower().split() if t}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union) if union else 0.0


class InMemoryVectorStore(VectorStore):
    """Minimal VectorStore useful for deterministic unit tests."""

    enabled: bool = True

    def __init__(self) -> None:
        # id -> (text, metadata)
        self._docs: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    def add_entry(
        self,
        entry_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._docs[entry_id] = (text, dict(metadata or {}))

    def query(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        scored = [
            {
                "id": doc_id,
                "text": doc_text,
                "score": _jaccard(text, doc_text),
                "metadata": dict(meta),
            }
            for doc_id, (doc_text, meta) in self._docs.items()
        ]
        scored.sort(key=lambda h: h["score"], reverse=True)
        return scored[: max(0, top_k)]

    # Convenience helpers not part of the interface; useful only in tests.
    def clear(self) -> None:
        self._docs.clear()

    def all_ids(self) -> List[str]:
        return list(self._docs.keys())
