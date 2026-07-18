from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from .base import RetrievalHit, VectorStore

_CHROMA_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    _CHROMA_AVAILABLE = False
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    logging.warning(f"Chroma import failed. Error: {type(e).__name__}")


_DEFAULT_NAMESPACE = "default"
_COLLECTION_PREFIX = "ns__"
# Chroma collection names must be 3-63 chars, [a-zA-Z0-9._-], start/end alnum.
_COLLECTION_SAFE = re.compile(r"[^a-zA-Z0-9._-]")


def _collection_name(namespace: str) -> str:
    """Sanitize a namespace into a Chroma-safe collection name."""
    ns = namespace or _DEFAULT_NAMESPACE
    safe = _COLLECTION_SAFE.sub("_", ns)
    return f"{_COLLECTION_PREFIX}{safe}"[:63]


class ChromaStore(VectorStore):
    """
    Chroma-backed local vector store.

    Namespaces are implemented as *separate collections* (`ns__{namespace}`)
    rather than metadata filters. Rationale: Chroma metadata filters force a
    full-collection scan per query; per-collection isolation scales better
    and matches Pinecone's per-namespace mental model.
    """

    def __init__(self, default_namespace: str = _DEFAULT_NAMESPACE) -> None:
        if not _CHROMA_AVAILABLE or chromadb is None:
            raise RuntimeError(
                "Chroma backend selected but chromadb is not available.\n"
                "Install optional dependencies: make setup-full"
            )

        # Default path aligns with upgrade 02's ./storage/ home.
        # CHROMA_PERSIST_DIR still honored for migrations.
        self._persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./storage/chroma")
        self._default_namespace = default_namespace or _DEFAULT_NAMESPACE

        try:
            os.makedirs(self._persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            logging.info(f"ChromaStore initialized at {self._persist_dir}")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaStore: {e}")
            raise RuntimeError(f"Failed to initialize ChromaStore: {e}") from e

    # ── VectorStore properties ────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return True

    @property
    def backend_name(self) -> str:
        return "chroma"

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _resolve_namespace(self, namespace: Optional[str]) -> str:
        return namespace or self._default_namespace

    def _get_collection(self, namespace: Optional[str]):
        ns = self._resolve_namespace(namespace)
        return self._client.get_or_create_collection(
            name=_collection_name(ns),
            metadata={"description": f"journal_entries (ns={ns})"},
        )

    @staticmethod
    def _distance_to_score(distance: Any) -> float:
        try:
            d = float(distance)
        except (TypeError, ValueError):
            return 0.0
        return 1.0 / (1.0 + d)

    # ── VectorStore methods ───────────────────────────────────────────────────
    def add_entry(
        self,
        entry_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        namespace: Optional[str] = None,
    ) -> bool:
        try:
            coll = self._get_collection(namespace)
            meta = dict(metadata or {})
            meta.setdefault("namespace", self._resolve_namespace(namespace))
            coll.add(documents=[text], ids=[entry_id], metadatas=[meta])
            return True
        except Exception as e:
            logging.error(f"Chroma add_entry failed: {e}")
            return False

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
        try:
            coll = self._get_collection(namespace)
            n_results = max(1, min(top_k, 10))
            kwargs: Dict[str, Any] = {
                "query_texts": [text],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if filter_metadata:
                kwargs["where"] = dict(filter_metadata)
            results = coll.query(**kwargs)
        except Exception as e:
            logging.error(f"Chroma query failed: {e}")
            return []

        docs = (results.get("documents") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []
        distances = (results.get("distances") or [[]])[0] or []
        ids = (results.get("ids") or [[]])[0] or []

        hits: List[RetrievalHit] = []
        for _id, doc, meta, dist in zip(ids, docs, metadatas, distances):
            hits.append(
                RetrievalHit(
                    id=_id,
                    text=doc or "",
                    score=self._distance_to_score(dist),
                    metadata=dict(meta or {}),
                )
            )
        return hits[:top_k]

    def delete_entry(
        self,
        entry_id: str,
        *,
        namespace: Optional[str] = None,
    ) -> None:
        try:
            coll = self._get_collection(namespace)
            coll.delete(ids=[entry_id])
        except Exception as e:
            logging.error(f"Chroma delete_entry failed: {e}")

    def clear_namespace(self, namespace: str) -> None:
        name = _collection_name(namespace or self._default_namespace)
        try:
            self._client.delete_collection(name=name)
        except Exception as e:
            # delete_collection raises if the collection doesn't exist;
            # callers treat clear_namespace as idempotent.
            logging.debug(f"Chroma clear_namespace({name}) no-op: {e}")

    def healthcheck(self) -> bool:
        try:
            self._client.list_collections()
            return True
        except Exception as e:
            logging.warning(f"Chroma healthcheck failed: {e}")
            return False
