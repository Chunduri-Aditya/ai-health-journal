from __future__ import annotations

import logging
from typing import Optional

from .base import VectorStore
from config import load_config


def get_vector_store(namespace: Optional[str] = None) -> VectorStore:
    """
    Return a concrete VectorStore implementation based on VECTOR_BACKEND.

    VECTOR_BACKEND=chroma   -> local Chroma (requires chromadb)
    VECTOR_BACKEND=pinecone -> Pinecone + LangChain integration (requires optional deps)
    VECTOR_BACKEND=none     -> returns a no-op store (retrieval disabled)

    The optional `namespace` is forwarded to the backend when supported
    (used for per-user or per-session isolation).
    """
    cfg = load_config()
    backend = cfg.vector_backend
    ns = namespace or cfg.rag_namespace_fixed

    if backend == "none":
        # Return a no-op store when retrieval is disabled
        class NoOpStore(VectorStore):
            def add_entry(self, entry_id: str, text: str, metadata=None):
                pass
            def query(self, text: str, top_k: int = 3):
                return []
        logging.info("Vector backend disabled (VECTOR_BACKEND=none).")
        return NoOpStore()

    if backend == "pinecone":
        if not cfg.allow_cloud_vectorstore:
            raise RuntimeError(
                "cloud_vectorstore_not_enabled: set ALLOW_CLOUD_VECTORSTORE=true to enable Pinecone."
            )
        try:
            from .pinecone_store import PineconeStore
        except ImportError:
            raise RuntimeError(
                "Pinecone backend selected but required packages are not available.\n"
                "Install optional dependencies: make setup-full\n"
                "Or: pip install -r requirements-optional.txt"
            )
        logging.info(f"Using Pinecone vector backend via LangChain (namespace={ns}).")
        return PineconeStore(namespace=ns)

    if backend == "chroma":
        try:
            from .chroma_store import ChromaStore
        except ImportError:
            raise RuntimeError(
                "Chroma backend selected but chromadb is not available.\n"
                "Install optional dependencies: make setup-full\n"
                "Or: pip install -r requirements-optional.txt"
            )
        logging.info(f"Using Chroma (local) vector backend (collection={ns}).")
        return ChromaStore(namespace=ns)

    logging.warning(f"Unknown VECTOR_BACKEND='{backend}', falling back to 'none' (disabled).")
    # Return no-op store for unknown backends
    class NoOpStore(VectorStore):
        def add_entry(self, entry_id: str, text: str, metadata=None):
            pass
        def query(self, text: str, top_k: int = 3):
            return []
    return NoOpStore()



