"""
Contract tests for the VectorStore interface (upgrade 03).

These exercise the widened surface: RetrievalHit return type, namespace
isolation, filter_metadata, delete_entry, clear_namespace, healthcheck.
They run against the in-memory fake here; ChromaStore gets an opt-in
integration variant in tests/vector_store/test_chroma_integration.py
(marked @pytest.mark.integration).
"""

from __future__ import annotations

import pytest

from vector_store.base import RetrievalHit, format_hits_as_context
from tests.vector_store.fakes import InMemoryVectorStore


@pytest.fixture
def store() -> InMemoryVectorStore:
    s = InMemoryVectorStore()
    s.add_entry("e1", "I argued with my friend again today", {"kind": "entry"})
    s.add_entry("e2", "The garden is in full bloom this week", {"kind": "entry"})
    s.add_entry("e3", "I keep procrastinating on the report", {"kind": "entry"})
    return s


class TestInterfaceShape:
    def test_query_returns_retrieval_hits(self, store):
        hits = store.query("anything", top_k=3)
        assert all(isinstance(h, RetrievalHit) for h in hits)
        for h in hits:
            assert isinstance(h.id, str)
            assert isinstance(h.text, str)
            assert isinstance(h.score, float)
            assert isinstance(h.metadata, dict)

    def test_retrieval_hit_to_dict_roundtrip(self):
        h = RetrievalHit(id="x", text="t", score=0.5, metadata={"k": 1})
        d = h.to_dict()
        assert d == {"id": "x", "text": "t", "score": 0.5, "metadata": {"k": 1}}
        # The returned dict is a copy of metadata, not an alias.
        d["metadata"]["k"] = 2
        assert h.metadata["k"] == 1


class TestAddAndQuery:
    def test_add_and_query_returns_hits(self, store):
        hits = store.query("argued with a friend", top_k=3)
        assert len(hits) == 3

    def test_ranking_prefers_higher_overlap(self, store):
        hits = store.query("argued with my friend", top_k=3)
        assert hits[0].id == "e1"

    def test_top_k_bounds_results(self, store):
        assert len(store.query("anything", top_k=1)) == 1
        assert len(store.query("anything", top_k=0)) == 0

    def test_metadata_is_preserved(self, store):
        hits = store.query("garden bloom", top_k=1)
        assert hits[0].metadata["kind"] == "entry"

    def test_adding_same_id_overwrites(self, store):
        store.add_entry("e1", "updated text here", {"kind": "entry", "v": 2})
        hits = [h for h in store.query("updated text", top_k=3) if h.id == "e1"]
        assert len(hits) == 1
        assert hits[0].text == "updated text here"
        assert hits[0].metadata["v"] == 2


class TestNamespaceIsolation:
    def test_two_namespaces_do_not_leak(self):
        s = InMemoryVectorStore()
        s.add_entry("a1", "alpha beta gamma", namespace="session:A")
        s.add_entry("b1", "delta epsilon zeta", namespace="session:B")

        a_hits = s.query("alpha beta", top_k=5, namespace="session:A")
        b_hits = s.query("alpha beta", top_k=5, namespace="session:B")

        assert [h.id for h in a_hits] == ["a1"]
        assert [h.id for h in b_hits] == ["b1"]

    def test_query_in_empty_namespace_is_empty(self):
        s = InMemoryVectorStore()
        s.add_entry("a1", "alpha", namespace="session:A")
        assert s.query("alpha", namespace="session:B") == []

    def test_default_namespace_is_used_when_unspecified(self):
        s = InMemoryVectorStore(default_namespace="ns-default")
        s.add_entry("x", "only doc")
        assert [h.id for h in s.query("only", namespace="ns-default")] == ["x"]
        assert s.query("only", namespace="other") == []


class TestFilterMetadata:
    def test_filter_narrows_results(self):
        s = InMemoryVectorStore()
        s.add_entry("a", "shared tokens here", {"kind": "entry"})
        s.add_entry("b", "shared tokens here", {"kind": "insight"})
        hits = s.query("shared tokens", top_k=5, filter_metadata={"kind": "entry"})
        assert [h.id for h in hits] == ["a"]


class TestDeleteAndClear:
    def test_delete_entry_removes_single_id(self, store):
        store.delete_entry("e1")
        assert "e1" not in {h.id for h in store.query("argued friend", top_k=5)}

    def test_delete_missing_id_is_no_op(self, store):
        store.delete_entry("never-added")
        assert len(store.query("anything", top_k=5)) == 3

    def test_clear_namespace_drops_all_entries(self):
        s = InMemoryVectorStore()
        s.add_entry("a", "x y z", namespace="session:A")
        s.add_entry("b", "p q r", namespace="session:A")
        s.clear_namespace("session:A")
        assert s.query("x y z", top_k=5, namespace="session:A") == []

    def test_clear_namespace_does_not_affect_siblings(self):
        s = InMemoryVectorStore()
        s.add_entry("a", "x y z", namespace="session:A")
        s.add_entry("b", "p q r", namespace="session:B")
        s.clear_namespace("session:A")
        assert [h.id for h in s.query("p q r", namespace="session:B")] == ["b"]


class TestHealthcheck:
    def test_default_reports_healthy(self):
        assert InMemoryVectorStore().healthcheck() is True

    def test_unhealthy_mode_returns_false(self):
        s = InMemoryVectorStore()
        s.set_healthy(False)
        assert s.healthcheck() is False


class TestDeterminism:
    def test_two_instances_score_identically(self):
        a = InMemoryVectorStore()
        b = InMemoryVectorStore()
        for s in (a, b):
            s.add_entry("x", "alpha beta gamma")
            s.add_entry("y", "gamma delta epsilon")
        qa = a.query("beta gamma", top_k=2)
        qb = b.query("beta gamma", top_k=2)
        assert [h.id for h in qa] == [h.id for h in qb]
        for ha, hb in zip(qa, qb):
            assert ha.score == pytest.approx(hb.score)


class TestFormatHelper:
    def test_format_hits_as_context_empty_returns_empty_string(self):
        assert format_hits_as_context([]) == ""

    def test_format_hits_as_context_includes_each_hit_text(self):
        hits = [
            RetrievalHit(id="a", text="first doc", score=0.9, metadata={"created_at": "t1"}),
            RetrievalHit(id="b", text="second doc", score=0.5, metadata={}),
        ]
        out = format_hits_as_context(hits)
        assert "first doc" in out
        assert "second doc" in out
        assert "t1" in out
