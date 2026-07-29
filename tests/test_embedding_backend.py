"""Tests for the configurable embedding backend and its mismatch guard.

Deliberately Ollama-free: the mismatch is provoked with a local stub embedding
function rather than a live nomic model, so these run offline in the normal
suite instead of behind the `integration` marker. Chroma's own
DefaultEmbeddingFunction cannot serve that role, since it is precisely what a
default-created collection already recorded.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass

import pytest

from vector_store.embeddings import (
    DEFAULT_BACKEND_DIMENSION,
    EmbeddingBackendMismatch,
    build_embedding_function,
    expected_dimension,
)


@dataclass
class _Cfg:
    embedding_backend: str = "default"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_embed_url: str = "http://localhost:11434"


class TestBuildEmbeddingFunction:
    def test_default_backend_returns_none(self):
        """None, not an explicitly-constructed default.

        Chroma records the embedding function on each collection, so passing an
        explicit default where none was recorded triggers the same mismatch error
        as a real backend change. Returning None keeps existing stores readable.
        """
        assert build_embedding_function(_Cfg(embedding_backend="default")) is None

    def test_unknown_backend_falls_back_to_default(self):
        assert build_embedding_function(_Cfg(embedding_backend="wat")) is None

    def test_missing_attributes_do_not_crash(self):
        """A config object predating these fields must still work."""

        class Bare:
            pass

        assert build_embedding_function(Bare()) is None


class TestExpectedDimension:
    def test_default_backend(self):
        assert expected_dimension("default", "") == DEFAULT_BACKEND_DIMENSION

    def test_known_model_ignores_tag(self):
        assert expected_dimension("ollama", "nomic-embed-text:latest") == 768

    def test_unknown_model_returns_none(self):
        assert expected_dimension("ollama", "some-future-embedder") is None


class _StubEmbeddingFunction:
    """A deliberately different embedding function, needing no network.

    Chroma's own DefaultEmbeddingFunction cannot be used to provoke the
    mismatch: it is exactly what a default-created collection already recorded,
    so passing it is a match, not a switch. This stub stands in for "some other
    backend" (nomic, or anything else) without requiring a running daemon.
    """

    @staticmethod
    def name() -> str:
        return "stub_ef"

    @staticmethod
    def build_from_config(config):
        return _StubEmbeddingFunction()

    def get_config(self):
        return {}

    def __call__(self, input):
        return [[float(len(t) % 7), 1.0, 2.0] for t in input]


class TestMismatchGuard:
    """The guard that makes a backend switch safe rather than corrupting."""

    def _store_with_default_entries(self, path):
        from vector_store.chroma_store import ChromaStore

        store = ChromaStore(default_namespace="guardtest")
        store.add_entry(entry_id="e1", text="felt anxious about work", namespace="guardtest")
        return store

    def test_switching_embedder_raises_actionable_error(self, monkeypatch):
        pytest.importorskip("chromadb")

        from vector_store.chroma_store import ChromaStore

        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setenv("CHROMA_PERSIST_DIR", tmp)
            self._store_with_default_entries(tmp)

            # Reopen the same collection with a *different* embedding function.
            switched = ChromaStore(
                default_namespace="guardtest",
                embedding_function=_StubEmbeddingFunction(),
            )
            with pytest.raises(EmbeddingBackendMismatch) as excinfo:
                switched.query("anxious", top_k=1, namespace="guardtest")

            message = str(excinfo.value)
            assert "migrate_embeddings.py" in message, "error must name the fix"
            assert "EMBEDDING_BACKEND=default" in message, "error must name the escape hatch"

    def test_mismatch_is_not_swallowed_into_a_false_return(self, monkeypatch):
        """add_entry returns False for ordinary write failures but must RAISE here.

        Returning False would present a fixable misconfiguration as routine data
        loss, silently, on every entry the user writes.
        """
        pytest.importorskip("chromadb")

        from vector_store.chroma_store import ChromaStore

        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setenv("CHROMA_PERSIST_DIR", tmp)
            self._store_with_default_entries(tmp)

            switched = ChromaStore(
                default_namespace="guardtest",
                embedding_function=_StubEmbeddingFunction(),
            )
            with pytest.raises(EmbeddingBackendMismatch):
                switched.add_entry(entry_id="e2", text="another entry", namespace="guardtest")

    def test_same_backend_still_works(self, monkeypatch):
        """Regression guard: the default path must be unchanged by all of this."""
        pytest.importorskip("chromadb")
        from vector_store.chroma_store import ChromaStore

        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setenv("CHROMA_PERSIST_DIR", tmp)
            self._store_with_default_entries(tmp)

            reopened = ChromaStore(default_namespace="guardtest")
            hits = reopened.query("anxious", top_k=1, namespace="guardtest")
            assert [h.id for h in hits] == ["e1"]
