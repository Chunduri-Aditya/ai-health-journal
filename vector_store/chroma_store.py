from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .base import VectorStore

# Lazy import: chromadb is in requirements-optional.txt
_CHROMA_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    # chromadb not installed - this is expected for core-only installs
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    _CHROMA_AVAILABLE = False
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    logging.warning(
        f"Chroma import failed. Error: {type(e).__name__}"
    )


class ChromaStore(VectorStore):
    """Chroma-based local vector store, mirroring existing RAG behavior."""

    def __init__(self, namespace: str = "journal_entries") -> None:
        if not _CHROMA_AVAILABLE or chromadb is None:
            raise RuntimeError(
                "Chroma backend selected but chromadb is not available.\n"
                "Install optional dependencies: make setup-full\n"
                "Or: pip install -r requirements-optional.txt"
            )

        self.enabled: bool = True
        self._client = None
        self._collection = None
        self._namespace = namespace

        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./rag_store")

        try:
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=namespace,
                metadata={"description": f"Journal entries and insights ({namespace})"},
            )
            logging.info(f"ChromaStore initialized at {persist_dir}")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaStore: {e}")
            raise RuntimeError(f"Failed to initialize ChromaStore: {e}")

    def add_entry(self, entry_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or not self._collection:
            return

        try:
            meta = metadata or {}
            self._collection.add(documents=[text], ids=[entry_id], metadatas=[meta])
        except Exception as e:
            logging.error(f"Failed to add entry to ChromaStore: {e}")

    def query(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.enabled or not self._collection:
            return []

        try:
            n_results = max(1, min(top_k, 10))
            results = self._collection.query(
                query_texts=[text],
                n_results=n_results,
                include=["documents", "metadatas", "distances", "ids"],
            )
        except Exception as e:
            logging.error(f"Failed to query ChromaStore: {e}")
            return []

        docs = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        ids = results.get("ids", [[]])[0] if results.get("ids") else []

        out: List[Dict[str, Any]] = []
        for doc, meta, dist, _id in zip(docs, metadatas, distances, ids):
            # Chroma returns a distance; convert to a similarity-ish score for interpretability.
            try:
                score = float(1.0 / (1.0 + float(dist)))
            except Exception:
                score = 0.0
            out.append(
                {
                    "id": _id,
                    "text": doc,
                    "metadata": meta or {},
                    "score": score,
                }
            )
        return out

