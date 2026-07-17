from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .base import RetrievalHit, VectorStore

_PINECONE_AVAILABLE = False
try:
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore
    from sentence_transformers import SentenceTransformer
    from privacy import local_text_cache
    _PINECONE_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    Pinecone = None  # type: ignore
    PineconeVectorStore = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    local_text_cache = None  # type: ignore


class _SentenceTransformerEmbeddings:
    """Minimal LangChain-style embeddings wrapper around SentenceTransformer."""

    def __init__(self, model: "SentenceTransformer") -> None:
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


class PineconeStore(VectorStore):
    """
    Pinecone-backed vector store using LangChain integration.

    Privacy defaults:
      - Uses local sentence-transformer embeddings.
      - By default does NOT store raw document text in Pinecone
        (set PINECONE_STORE_TEXT=true to opt in).

    Namespace handling:
      - `add_entry` / `query` accept an explicit `namespace=` kwarg.
      - Falls back to the configured default namespace when omitted.
    """

    def __init__(self, default_namespace: Optional[str] = None) -> None:
        if not _PINECONE_AVAILABLE:
            raise RuntimeError(
                "Pinecone backend selected but required packages are not available.\n"
                "Install optional dependencies: make setup-full"
            )

        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX")
        if not api_key or not index_name:
            raise RuntimeError(
                "Pinecone backend selected but PINECONE_API_KEY or PINECONE_INDEX "
                "is not set. Configure your environment."
            )

        self._default_namespace = (
            default_namespace
            or os.getenv("PINECONE_NAMESPACE")
            or "ai-health-journal"
        )
        embed_model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        self._embedder = SentenceTransformer(embed_model_name)
        dim = getattr(self._embedder, "get_sentence_embedding_dimension", lambda: None)()
        if dim is not None:
            env_dim = int(os.getenv("PINECONE_DIM", str(dim)))
            if env_dim != dim:
                raise RuntimeError(
                    f"Pinecone dim mismatch: EMBED_MODEL '{embed_model_name}' gives dim={dim}, "
                    f"but PINECONE_DIM={env_dim}. Update .env or recreate the index "
                    f"via tools/pinecone_bootstrap.py."
                )

        self._store_text: bool = os.getenv("PINECONE_STORE_TEXT", "false").lower() == "true"

        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)
        self._embeddings = _SentenceTransformerEmbeddings(self._embedder)
        self._vs_cache: Dict[str, "PineconeVectorStore"] = {}

    # ── VectorStore properties ────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return True

    @property
    def backend_name(self) -> str:
        return "pinecone"

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _resolve_namespace(self, namespace: Optional[str]) -> str:
        return namespace or self._default_namespace

    def _vs_for(self, namespace: Optional[str]) -> "PineconeVectorStore":
        ns = self._resolve_namespace(namespace)
        cached = self._vs_cache.get(ns)
        if cached is not None:
            return cached
        vs = PineconeVectorStore(
            index=self._index,
            embedding=self._embeddings,
            namespace=ns,
        )
        self._vs_cache[ns] = vs
        return vs

    # ── VectorStore methods ───────────────────────────────────────────────────
    def add_entry(
        self,
        entry_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        namespace: Optional[str] = None,
    ) -> None:
        ns = self._resolve_namespace(namespace)
        meta = dict(metadata or {})
        meta.setdefault("source_id", entry_id)
        meta.setdefault("namespace", ns)
        doc_text = text if self._store_text else ""
        try:
            self._vs_for(ns).add_texts([doc_text], metadatas=[meta], ids=[entry_id])
        except Exception as e:
            logging.error(f"Pinecone add_entry failed: {e}")

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
        ns = self._resolve_namespace(namespace)
        try:
            kwargs: Dict[str, Any] = {"k": top_k}
            if filter_metadata:
                kwargs["filter"] = dict(filter_metadata)
            results = self._vs_for(ns).similarity_search_with_score(text, **kwargs)
        except Exception as e:
            logging.error(f"Pinecone query failed: {e}")
            return []

        hits: List[RetrievalHit] = []
        for doc, score in results:
            meta = dict(doc.metadata or {})
            source_id = meta.get("source_id") or ""
            page = doc.page_content or ""
            if not page and source_id and local_text_cache is not None:
                cached = local_text_cache.get(meta.get("namespace", ns), source_id)
                if cached:
                    page = cached.get("text", "") or ""
            hits.append(
                RetrievalHit(
                    id=source_id,
                    text=page,
                    score=float(score),
                    metadata=meta,
                )
            )
        return hits

    def delete_entry(
        self,
        entry_id: str,
        *,
        namespace: Optional[str] = None,
    ) -> None:
        ns = self._resolve_namespace(namespace)
        try:
            self._index.delete(ids=[entry_id], namespace=ns)
        except Exception as e:
            logging.error(f"Pinecone delete_entry failed: {e}")

    def clear_namespace(self, namespace: str) -> None:
        ns = namespace or self._default_namespace
        try:
            self._index.delete(delete_all=True, namespace=ns)
            self._vs_cache.pop(ns, None)
        except Exception as e:
            logging.debug(f"Pinecone clear_namespace({ns}) failed: {e}")

    def healthcheck(self) -> bool:
        try:
            self._index.describe_index_stats()
            return True
        except Exception as e:
            logging.warning(f"Pinecone healthcheck failed: {e}")
            return False
