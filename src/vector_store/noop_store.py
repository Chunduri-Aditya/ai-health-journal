"""
No-op vector store used when retrieval is disabled or when an optional
backend dependency is missing. All writes are dropped and all queries
return an empty list, so callers can unconditionally call add_entry /
query without guards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import RetrievalHit, VectorStore


class NoOpStore(VectorStore):
    @property
    def enabled(self) -> bool:
        return False

    @property
    def backend_name(self) -> str:
        return "none"

    def add_entry(
        self,
        entry_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        namespace: Optional[str] = None,
    ) -> bool:
        return True

    def query(
        self,
        text: str,
        *,
        top_k: int = 3,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalHit]:
        return []

    def delete_entry(
        self,
        entry_id: str,
        *,
        namespace: Optional[str] = None,
    ) -> None:
        return None

    def clear_namespace(self, namespace: str) -> None:
        return None

    def healthcheck(self) -> bool:
        return True
