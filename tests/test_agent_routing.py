from __future__ import annotations

from src.agent.confirmation import is_confirmation, is_denial
from src.agent.graph import _strip_write_directive, route_intent
from src.agent.tools import tool_apply_write, tool_propose_write, tool_retrieve
from tests.vector_store.fakes import InMemoryVectorStore
from src.vector_store.noop_store import NoOpStore


class TestConfirmation:
    def test_confirm_phrases(self):
        assert is_confirmation("yes")
        assert is_confirmation("confirm")
        assert is_confirmation("approve")

    def test_ok_not_confirmation(self):
        """'ok' and 'okay' should NOT be confirmation phrases (removed for Week 3)."""
        assert not is_confirmation("ok")
        assert not is_confirmation("okay")

    def test_deny_phrases(self):
        assert is_denial("no")
        assert is_denial("cancel")


class TestRouteIntent:
    def test_retrieve(self):
        state = {"messages": [{"role": "user", "content": "find entries about work"}]}
        assert route_intent(state) == "retrieve"

    def test_metadata(self):
        state = {"messages": [{"role": "user", "content": "list entries metadata"}]}
        assert route_intent(state) == "metadata"

    def test_write(self):
        state = {"messages": [{"role": "user", "content": "save this: I felt calm today"}]}
        assert route_intent(state) == "propose_write"

    def test_confirm_gate_when_pending(self):
        state = {
            "messages": [{"role": "user", "content": "confirm"}],
            "awaiting_confirmation": True,
            "pending_write": {"entry_id": "x"},
        }
        assert route_intent(state) == "confirm_gate"

    def test_respond_default(self):
        state = {"messages": [{"role": "user", "content": "how are you"}]}
        assert route_intent(state) == "respond"


class TestStripWriteDirective:
    def test_save_this_colon(self):
        assert _strip_write_directive("save this: evening walk") == "evening walk"

    def test_write_colon(self):
        assert _strip_write_directive("write: calm day") == "calm day"

    def test_bare_text_unchanged(self):
        assert _strip_write_directive("just a journal thought") == "just a journal thought"

    def test_please_remember_this(self):
        assert _strip_write_directive("please remember this: felt peaceful") == "felt peaceful"


class TestApplyWrite:
    def test_noop_store_reports_failed(self):
        pending = {
            "entry_id": "e1",
            "text": "evening walk",
            "metadata": {},
            "namespace": "ns",
        }
        out = tool_apply_write(NoOpStore(), pending)
        assert out["status"] == "failed"
        assert out["reason"] == "retrieval_disabled"

    def test_memory_store_applies(self):
        store = InMemoryVectorStore(default_namespace="ns")
        pending = {
            "entry_id": "e1",
            "text": "evening walk felt peaceful",
            "metadata": {"kind": "entry"},
            "namespace": "ns",
        }
        out = tool_apply_write(store, pending)
        assert out["status"] == "applied"
        hits = store.query("peaceful", namespace="ns", top_k=1)
        assert hits and hits[0].id == "e1"


class TestToolsAgainstFakeStore:
    def test_retrieve_returns_hits(self):
        store = InMemoryVectorStore()
        store.add_entry("e1", "I argued with my friend", namespace="ns")
        out = tool_retrieve(store, "argued friend", namespace="ns", top_k=1)
        assert out["tool"] == "retrieve_journal"
        assert out["hits"][0]["id"] == "e1"

    def test_propose_write_does_not_mutate(self):
        store = InMemoryVectorStore()
        out = tool_propose_write(text="new entry", namespace="ns")
        assert out["status"] == "awaiting_confirmation"
        assert store.query("new entry", namespace="ns") == []
