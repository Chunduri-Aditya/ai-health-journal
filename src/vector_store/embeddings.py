"""Embedding function selection for the local vector store.

Why this is configurable rather than hardcoded
-----------------------------------------------
Chroma's default embedder is ONNX all-MiniLM-L6-v2 (384-dim). Measured on the
diagnostic corpus (docs/IMPROVEMENTS.md section 10), it fails an entire class of
journaling queries: a positive entry retrieves the user's worst entries, because
the model encodes topic and not emotional valence. `nomic-embed-text` (768-dim,
served by the Ollama daemon this project already requires) scores better or
equal in every trap category on both corpora, and takes valence_flip recall from
0.667 to 1.000.

Backends
--------
`default` - Chroma's bundled ONNX MiniLM. No extra setup, 384-dim.
`ollama`  - any embedding model served by the local Ollama daemon.
            Requires `ollama pull <model>` first. Stays fully local.

Dimensions differ between backends, and Chroma refuses to mix embedding
functions within one collection. That refusal is the safety net: switching
backends against an existing store raises rather than corrupting it. See
scripts/migrate_embeddings.py for the re-embedding path.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Dimensions are recorded so a mismatch can be reported in human terms rather
# than as a Chroma internal error. Not exhaustive; unknown models report None.
KNOWN_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}

DEFAULT_BACKEND_DIMENSION = 384  # Chroma's bundled ONNX MiniLM.


def expected_dimension(backend: str, model: str) -> Optional[int]:
    """Best-effort embedding width for a backend/model pair, or None if unknown."""
    if backend == "default":
        return DEFAULT_BACKEND_DIMENSION
    base = (model or "").split(":")[0]
    return KNOWN_DIMENSIONS.get(base)


def build_embedding_function(cfg) -> Optional[Any]:
    """Return a Chroma embedding function, or None to use Chroma's default.

    Returning None rather than explicitly constructing the default keeps
    existing collections readable: Chroma stores the embedding-function config
    on the collection, and passing an explicitly-constructed default where none
    was recorded triggers the same mismatch error as a real backend change.
    """
    backend = (getattr(cfg, "embedding_backend", "default") or "default").lower()

    if backend == "default":
        return None

    if backend != "ollama":
        logger.warning(
            "Unknown EMBEDDING_BACKEND=%r; falling back to Chroma's default embedder.",
            backend,
        )
        return None

    model = getattr(cfg, "ollama_embed_model", "nomic-embed-text")
    url = getattr(cfg, "ollama_embed_url", "http://localhost:11434")

    try:
        from chromadb.utils import embedding_functions as chroma_ef
    except ImportError:
        logger.error("chromadb is unavailable; cannot build an Ollama embedder.")
        return None

    try:
        fn = chroma_ef.OllamaEmbeddingFunction(url=url, model_name=model)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        logger.error(
            "Could not construct the Ollama embedder (model=%s): %s. "
            "Falling back to Chroma's default. Run `ollama pull %s` and restart.",
            model,
            type(exc).__name__,
            model,
        )
        return None

    logger.info("Using Ollama embedding backend (model=%s, url=%s).", model, url)
    return fn


class EmbeddingBackendMismatch(RuntimeError):
    """Raised when a store's collections were written by a different embedder.

    Carries a remediation message rather than leaking Chroma's internal error,
    because the only correct response is running the migration.
    """
