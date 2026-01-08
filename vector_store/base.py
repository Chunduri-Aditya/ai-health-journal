from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStore(ABC):
    """Abstract vector store interface so the app doesn't care about backend."""

    @abstractmethod
    def add_entry(self, entry_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Index a new text document with associated metadata."""
        raise NotImplementedError

    @abstractmethod
    def query(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Return a list of structured sources sorted by relevance.

        Each source dict MUST have:
          - id: str
          - text: str
          - score: float
          - metadata: dict
        """
        raise NotImplementedError

