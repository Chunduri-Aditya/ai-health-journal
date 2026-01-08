from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

try:
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore
    from sentence_transformers import SentenceTransformer
    from .base import VectorStore
    from privacy import local_text_cache
    _PINECONE_AVAILABLE = True
except Exception as e:  # pragma: no cover - import guard
    _PINECONE_AVAILABLE = False
    # Delay import errors until backend is actually selected
    Pinecone = None  # type: ignore
    PineconeVectorStore = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    VectorStore = object  # type: ignore
    local_text_cache = None  # type: ignore


class _SentenceTransformerEmbeddings:
    """Minimal LangChain-style embeddings wrapper around SentenceTransformer."""

    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


class PineconeStore(VectorStore):
    """
    Pinecone-based vector store using LangChain integration.

    Privacy defaults:
      - Uses local embeddings (SentenceTransformer)
      - By default, does NOT store raw document text in Pinecone
        (only embeddings + minimal metadata).
    """

    def __init__(self, namespace: str = "ai-health-journal") -> None:
        if not _PINECONE_AVAILABLE:
            raise RuntimeError(
                "Pinecone backend selected but required packages are not available.\n"
                "Install optional dependencies: make setup-full\n"
                "Or: pip install -r requirements-optional.txt"
            )
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX")
        self._namespace = namespace or os.getenv("PINECONE_NAMESPACE", "ai-health-journal")

        if not api_key or not index_name:
            raise RuntimeError(
                "Pinecone backend selected but PINECONE_API_KEY or PINECONE_INDEX "
                "is not set. Please configure your environment."
            )

        embed_model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        self._embedder = SentenceTransformer(embed_model_name)
        dim = getattr(self._embedder, "get_sentence_embedding_dimension", lambda: None)()
        if dim is not None:
            env_dim = int(os.getenv("PINECONE_DIM", str(dim)))
            if env_dim != dim:
                raise RuntimeError(
                    f"Pinecone dim mismatch: EMBED_MODEL '{embed_model_name}' gives dim={dim}, "
                    f"but PINECONE_DIM={env_dim}. Update your .env or recreate the index "
                    f"via tools/pinecone_bootstrap.py."
                )

        self.store_text: bool = os.getenv("PINECONE_STORE_TEXT", "false").lower() == "true"

        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)

        embeddings = _SentenceTransformerEmbeddings(self._embedder)

        # LangChain wrapper around the Pinecone index
        self._vs = PineconeVectorStore(index=self._index, embedding=embeddings, namespace=self._namespace)

    def add_entry(self, entry_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        # Respect privacy setting: optionally drop raw text
        doc_text = text if self.store_text else ""
        meta = metadata or {}
        # Record source id and namespace in metadata for later interpretability/cache
        meta.setdefault("source_id", entry_id)
        meta.setdefault("namespace", self._namespace)
        try:
            self._vs.add_texts([doc_text], metadatas=[meta], ids=[entry_id])
        except Exception as e:
            logging.error(f"Failed to add entry to PineconeStore: {e}")

    def query(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            results = self._vs.similarity_search_with_score(text, k=top_k)
        except Exception as e:
            logging.error(f"Failed to query PineconeStore: {e}")
            return []

        out: List[Dict[str, Any]] = []
        for doc, score in results:
            meta = doc.metadata or {}
            source_id = meta.get("source_id")
            # Fallback to local cache when text is empty (PINECONE_STORE_TEXT=false)
            page_content = doc.page_content or ""
            if not page_content and source_id:
                cached = local_text_cache.get(meta.get("namespace", self._namespace), source_id)
                if cached:
                    page_content = cached.get("text", "") or ""

            out.append(
                {
                    "id": source_id,
                    "text": page_content,
                    "metadata": meta,
                    "score": float(score),
                }
            )
        return out

