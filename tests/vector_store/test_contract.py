"""
Contract tests for the VectorStore interface.

Today these exercise the current `add_entry` / `query` surface against
the InMemoryVectorStore fake. Upgrade 03 will broaden the interface
(namespace, filter_metadata, delete_entry, clear_namespace,
healthcheck) and should add parametrized runs against ChromaStore
behind `@pytest.mark.integration`.
"""

from __future__ import annotations

from typing import List

import pytest

from tests.vector_store.fakes import InMemoryVectorStore


@pytest.fixture
def store() -> InMemoryVectorStore:
    s = InMemoryVectorStore()
    s.add_entry("e1", "I argued with my friend again today", {"kind": "entry"})
    s.add_entry("e2", "The garden is in full bloom this week", {"kind": "entry"})
    s.add_entry("e3", "I keep procrastinating on the report", {"kind": "entry"})
    return s


class TestAddAndQuery:
    def test_add_and_query_returns_hits(self, store):
        hits = store.query("argued with a friend", top_k=3)
        assert len(hits) == 3
        # All required fields present in the canonical shape.
        for h in hits:
            assert set(h.keys()) >= {"id", "text", "score", "metadata"}

    def test_ranking_prefers_higher_overlap(self, store):
        hits = store.query("argued with my friend", top_k=3)
        assert hits[0]["id"] == "e1"

    def test_top_k_bounds_results(self, store):
        assert len(store.query("anything", top_k=1)) == 1
        assert len(store.query("anything", top_k=0)) == 0

    def test_metadata_is_preserved(self, store):
        hits = store.query("garden bloom", top_k=1)
        assert hits[0]["metadata"]["kind"] == "entry"

    def test_adding_same_id_overwrites(self, store):
        store.add_entry("e1", "updated text here", {"kind": "entry", "v": 2})
        hits = [h for h in store.query("updated text", top_k=3) if h["id"] == "e1"]
        assert len(hits) == 1
        assert hits[0]["text"] == "updated text here"
        assert hits[0]["metadata"]["v"] == 2


class TestDeterminism:
    def test_two_instances_score_identically(self):
        a = InMemoryVectorStore()
        b = InMemoryVectorStore()
        for store in (a, b):
            store.add_entry("x", "alpha beta gamma", {})
            store.add_entry("y", "gamma delta epsilon", {})
        qa = a.query("beta gamma", top_k=2)
        qb = b.query("beta gamma", top_k=2)
        assert [h["id"] for h in qa] == [h["id"] for h in qb]
        for ha, hb in zip(qa, qb):
            assert ha["score"] == pytest.approx(hb["score"])


class TestEmptyStore:
    def test_query_empty_store_returns_empty_list(self):
        s = InMemoryVectorStore()
        assert s.query("anything", top_k=3) == []
