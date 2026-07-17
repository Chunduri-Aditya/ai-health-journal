from __future__ import annotations

import logging
from typing import Optional

from config import load_config

from .base import VectorStore
from .noop_store import NoOpStore


def get_vector_store(
    default_namespace: Optional[str] = None,
) -> VectorStore:
    """
    Return a concrete VectorStore based on config.

    Behavior:
      - If `RETRIEVAL_ENABLED=false`, returns NoOpStore regardless of backend.
      - Else, `VECTOR_BACKEND`:
          - `none`     -> NoOpStore
          - `chroma`   -> ChromaStore (requires chromadb)
          - `pinecone` -> PineconeStore (requires optional deps + cloud opt-in)
          - anything else -> logs a warning and returns NoOpStore

    `default_namespace` is a fallback used when a caller doesn't pass
    a per-call `namespace=` kwarg. It does not fix the namespace — each
    call may still supply its own.
    """
    cfg = load_config()
    ns = default_namespace or cfg.rag_namespace_fixed

    if not cfg.retrieval_enabled:
        logging.info("Retrieval disabled (RETRIEVAL_ENABLED=false).")
        return NoOpStore()

    backend = (cfg.vector_backend or "none").lower()

    if backend == "none":
        logging.info("Vector backend disabled (VECTOR_BACKEND=none).")
        return NoOpStore()

    if backend == "chroma":
        try:
            from .chroma_store import ChromaStore
        except ImportError as e:
            logging.error(f"Chroma selected but unavailable; using noop. {e}")
            return NoOpStore()
        try:
            logging.info(f"Using Chroma vector backend (default namespace={ns}).")
            return ChromaStore(default_namespace=ns)
        except RuntimeError as e:
            logging.error(f"Chroma init failed; using noop. {e}")
            return NoOpStore()

    if backend == "pinecone":
        if not cfg.allow_cloud_vectorstore:
            raise RuntimeError(
                "cloud_vectorstore_not_enabled: set ALLOW_CLOUD_VECTORSTORE=true "
                "to enable Pinecone."
            )
        try:
            from .pinecone_store import PineconeStore
        except ImportError as e:
            raise RuntimeError(f"Pinecone backend unavailable: {e}") from e
        logging.info(f"Using Pinecone vector backend (default namespace={ns}).")
        return PineconeStore(default_namespace=ns)

    logging.warning(f"Unknown VECTOR_BACKEND={backend!r}; using noop.")
    return NoOpStore()
