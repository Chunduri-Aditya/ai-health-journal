from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RetrievalHit:
    """
    Single retrieval result.

    `score` is a relative ranking signal within one backend + query.
    It is NOT a normalized confidence portable across Chroma and Pinecone.
    Higher == more relevant within that result set.
    """

    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "metadata": dict(self.metadata),
        }


class VectorStore(ABC):
    """Abstract vector store interface so the app doesn't care about backend."""

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """
        True if this backend is wired up and writes/queries should be honored.
        The no-op store returns False so the caller can still hold a reference.
        """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier used in /ping and diagnostics, e.g. 'chroma' or 'none'."""

    @abstractmethod
    def add_entry(
        self,
        entry_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        namespace: Optional[str] = None,
    ) -> bool:
        """Index a new text document with associated metadata.

        Returns True on success, False if the write failed. Implementations
        must not raise for backend-level failures (a RAG indexing failure
        should not break the caller's primary response) -- they catch, log,
        and return False so the caller can at least detect and log the
        failure, rather than the write being silently indistinguishable from
        success.
        """

    @abstractmethod
    def query(
        self,
        text: str,
        *,
        top_k: int = 3,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalHit]:
        """Return retrieval hits sorted by descending `score`."""

    @abstractmethod
    def delete_entry(
        self,
        entry_id: str,
        *,
        namespace: Optional[str] = None,
    ) -> None:
        """Remove a single entry by id. Missing ids are a no-op."""

    @abstractmethod
    def clear_namespace(self, namespace: str) -> None:
        """Drop every entry in the given namespace. Missing namespace is a no-op."""

    @abstractmethod
    def healthcheck(self) -> bool:
        """Non-raising probe. Returns True if the backend is reachable."""


def format_hits_as_context(hits: List[RetrievalHit]) -> str:
    """
    Render retrieval hits into the string shape the prompt templates expect.

    Kept here (rather than in a prompt module) so every consumer sees the same
    formatting whether it's app.py, evals/, or a future surface.
    """
    if not hits:
        return ""
    parts: List[str] = []
    for i, h in enumerate(hits, 1):
        ts = h.metadata.get("created_at") or h.metadata.get("timestamp") or ""
        header = f"[Retrieved Context {i}"
        if ts:
            header += f" | {ts}"
        header += "]"
        parts.append(f"{header}\n{h.text}\n")
    return "\n".join(parts)
