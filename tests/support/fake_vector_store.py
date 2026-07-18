"""In-memory VectorStore double with token-overlap ranking.

Fast enough for the default test suite (no embedding model, no chromadb) and
faithful to the parts the conversation loop depends on: per-namespace isolation,
id-based self-exclusion, deterministic descending-score ordering, and the
add/query/delete/clear contract. It does NOT model semantic similarity — that is
what the real-Chroma `slow` tests and `evals/rag_retrieval_eval.py` are for.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from vector_store.base import RetrievalHit, VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class InMemoryVectorStore(VectorStore):
    """Deterministic, network-free vector store for the fast conversation path."""

    def __init__(self) -> None:
        # namespace -> {entry_id: {"text": str, "metadata": dict}}
        self._ns: Dict[str, Dict[str, Dict[str, Any]]] = {}

    @property
    def enabled(self) -> bool:
        return True

    @property
    def backend_name(self) -> str:
        return "fake"

    def add_entry(
        self,
        entry_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        namespace: Optional[str] = None,
    ) -> bool:
        ns = namespace or ""
        self._ns.setdefault(ns, {})[entry_id] = {
            "text": text,
            "metadata": dict(metadata or {}),
        }
        return True

    def query(
        self,
        text: str,
        *,
        top_k: int = 3,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalHit]:
        ns = namespace or ""
        entries = self._ns.get(ns, {})
        q = _tokens(text)
        scored: List[RetrievalHit] = []
        for entry_id, rec in entries.items():
            score = _jaccard(q, _tokens(rec["text"]))
            if score <= 0.0:
                continue
            scored.append(
                RetrievalHit(
                    id=entry_id,
                    text=rec["text"],
                    score=score,
                    metadata=dict(rec["metadata"]),
                )
            )
        # Descending score; id as a stable tiebreaker so ordering is deterministic.
        scored.sort(key=lambda h: (h.score, h.id), reverse=True)
        return scored[:top_k]

    def delete_entry(self, entry_id: str, *, namespace: Optional[str] = None) -> None:
        self._ns.get(namespace or "", {}).pop(entry_id, None)

    def clear_namespace(self, namespace: str) -> None:
        self._ns.pop(namespace or "", None)

    def healthcheck(self) -> bool:
        return True
