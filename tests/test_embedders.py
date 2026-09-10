from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from src.vector_store.embedders import HashEmbedder, build_embedder

os.environ.setdefault("ENV", "test")
os.environ.setdefault("ALLOW_HASH_EMBEDDER", "true")


class TestHashEmbedder:
    def test_dimension(self):
        e = HashEmbedder(dimension=32)
        vec = e.embed_query("argued with a friend")
        assert len(vec) == 32

    def test_deterministic(self):
        e = HashEmbedder(dimension=64)
        assert e.embed_query("same") == e.embed_query("same")

    def test_different_texts_differ(self):
        e = HashEmbedder(dimension=64)
        assert e.embed_query("alpha") != e.embed_query("beta")

    def test_normalized(self):
        e = HashEmbedder(dimension=48)
        vec = e.embed_query("normalize me")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6


class TestBuildEmbedder:
    def test_hash_requires_allowance(self, monkeypatch):
        monkeypatch.setenv("ENV", "production")
        monkeypatch.delenv("ALLOW_HASH_EMBEDDER", raising=False)
        cfg = SimpleNamespace(embedding_backend="hash", embedding_dimension=16)
        with pytest.raises(RuntimeError, match="test-only"):
            build_embedder(cfg)

    def test_hash_alias_when_allowed(self, monkeypatch):
        monkeypatch.setenv("ALLOW_HASH_EMBEDDER", "true")
        cfg = SimpleNamespace(embedding_backend="hash", embedding_dimension=0)
        e = build_embedder(cfg)
        assert e.backend_name == "hash"

    def test_default_is_fastembed_or_raises_missing_dep(self, monkeypatch):
        monkeypatch.setenv("ENV", "dev")
        monkeypatch.delenv("ALLOW_HASH_EMBEDDER", raising=False)
        cfg = SimpleNamespace(embedding_backend="default", embedding_dimension=384)
        try:
            e = build_embedder(cfg)
            assert e.backend_name == "fastembed"
            assert e.dimension == 384
        except RuntimeError as err:
            assert "fastembed" in str(err).lower()
