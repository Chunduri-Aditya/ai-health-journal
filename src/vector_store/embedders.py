"""Backend-agnostic embedding functions returning float vectors.

Chroma's embedding helpers are Chroma-specific. pgvector (and any future
store) needs raw vectors. This module is the shared surface:

  - ``default`` / ``fastembed`` — ONNX MiniLM via fastembed (384-dim, local)
  - ``hash``     — deterministic bag-of-bytes vectors for unit tests only
                   (requires ALLOW_HASH_EMBEDDER=true or ENV=test)
  - ``ollama``   — local Ollama embedding models (nomic-embed-text, etc.)
  - ``openai``   — hosted OpenAI embeddings (cloud deploy path)

Dimensions must match the pgvector column / HNSW index. Changing backend or
model against an existing store requires re-ingestion.

GateGuard facts: callers are vector_store/factory.py, scripts/chunk_sweep.py,
tests/test_embedders.py. Existing file (not new). No data files. User instruction:
Implement the plan as specified, it is attached for your reference. Do NOT edit
the plan file itself.
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

KNOWN_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "hash": 384,
    "fastembed": 384,
}

_FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def _hash_embedder_allowed() -> bool:
    if os.getenv("ALLOW_HASH_EMBEDDER", "").lower() == "true":
        return True
    return (os.getenv("ENV") or "dev").strip().lower() in {"test"}


class HashEmbedder(Embedder):
    """Deterministic, dependency-free embedder for tests and offline smoke runs."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def backend_name(self) -> str:
        return "hash"

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        vals: List[float] = []
        seed = digest
        while len(vals) < self._dimension:
            seed = hashlib.sha256(seed).digest()
            for i in range(0, len(seed) - 3, 4):
                if len(vals) >= self._dimension:
                    break
                raw = struct.unpack_from(">I", seed, i)[0]
                vals.append((raw / 0xFFFFFFFF) * 2.0 - 1.0)
        norm = sum(v * v for v in vals) ** 0.5 or 1.0
        return [v / norm for v in vals]


class FastEmbedEmbedder(Embedder):
    """Local ONNX MiniLM embedder (semantic default for pgvector)."""

    def __init__(
        self,
        model: str = _FASTEMBED_MODEL,
        dimension: Optional[int] = None,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RuntimeError(
                "EMBEDDING_BACKEND=default/fastembed requires fastembed. "
                "Install: pip install -r requirements-service.txt"
            ) from e
        self._model_name = model
        self._model = TextEmbedding(model_name=model)
        self._dimension = int(dimension or KNOWN_DIMENSIONS.get(model, 384))

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def backend_name(self) -> str:
        return "fastembed"

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = [list(map(float, vec)) for vec in self._model.embed(list(texts))]
        if vectors and len(vectors[0]) != self._dimension:
            self._dimension = len(vectors[0])
        return vectors


class OllamaEmbedder(Embedder):
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: Optional[int] = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        base = model.split(":")[0]
        self._dimension = dimension or KNOWN_DIMENSIONS.get(base, 768)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def backend_name(self) -> str:
        return "ollama"

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        import requests

        try:
            resp = requests.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": list(texts)},
                timeout=120,
            )
            if resp.status_code == 200:
                payload = resp.json()
                vectors = payload.get("embeddings") or []
                if vectors:
                    out = [[float(x) for x in vec] for vec in vectors]
                    if out and len(out[0]) != self._dimension:
                        self._dimension = len(out[0])
                    return out
        except Exception as e:
            logger.debug("Ollama /api/embed failed (%s); falling back.", e)

        out: List[List[float]] = []
        for text in texts:
            resp = requests.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            vec = resp.json().get("embedding")
            if not isinstance(vec, list) or not vec:
                raise RuntimeError(f"Ollama returned empty embedding for model={self._model}")
            out.append([float(x) for x in vec])
        if out and len(out[0]) != self._dimension:
            logger.warning(
                "Ollama embedding dim %s != configured %s; updating to measured dim.",
                len(out[0]),
                self._dimension,
            )
            self._dimension = len(out[0])
        return out


class OpenAIEmbedder(Embedder):
    _client = None

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is required for EMBEDDING_BACKEND=openai")
        self._dimension = dimension or KNOWN_DIMENSIONS.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def backend_name(self) -> str:
        return "openai"

    def _get_client(self):
        from openai import OpenAI

        if OpenAIEmbedder._client is None:
            OpenAIEmbedder._client = OpenAI(api_key=self._api_key)
        return OpenAIEmbedder._client

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        client = self._get_client()
        resp = client.embeddings.create(model=self._model, input=list(texts))
        vectors = [list(item.embedding) for item in resp.data]
        if vectors and len(vectors[0]) != self._dimension:
            self._dimension = len(vectors[0])
        return vectors


def build_embedder(cfg) -> Embedder:
    """Construct an Embedder from app Config / env-backed dataclass."""
    backend = (getattr(cfg, "embedding_backend", "default") or "default").lower()
    dim = getattr(cfg, "embedding_dimension", None)
    dim_i = int(dim) if dim else None

    if backend == "hash":
        if not _hash_embedder_allowed():
            raise RuntimeError(
                "EMBEDDING_BACKEND=hash is test-only. "
                "Set ALLOW_HASH_EMBEDDER=true or ENV=test, "
                "or use EMBEDDING_BACKEND=default|fastembed|ollama|openai."
            )
        return HashEmbedder(dimension=int(dim_i or KNOWN_DIMENSIONS["hash"]))

    if backend in ("default", "fastembed"):
        return FastEmbedEmbedder(
            model=_FASTEMBED_MODEL,
            dimension=dim_i,
        )

    if backend == "ollama":
        return OllamaEmbedder(
            model=getattr(cfg, "ollama_embed_model", "nomic-embed-text"),
            base_url=getattr(cfg, "ollama_embed_url", "http://localhost:11434"),
            dimension=dim_i,
        )

    if backend == "openai":
        if not getattr(cfg, "allow_cloud_vectorstore", False):
            raise RuntimeError(
                "EMBEDDING_BACKEND=openai requires ALLOW_CLOUD_VECTORSTORE=true "
                "(same cloud gate as managed vector stores)."
            )
        return OpenAIEmbedder(
            model=getattr(cfg, "openai_embed_model", "text-embedding-3-small"),
            dimension=dim_i,
        )

    raise RuntimeError(
        f"Unknown EMBEDDING_BACKEND={backend!r}. "
        "Use default|fastembed|ollama|openai (or hash with ALLOW_HASH_EMBEDDER=true)."
    )
