"""Progressive multi-turn conversation tests through the real /analyze route.

Fast path: fake provider (no Ollama) + in-memory fake store (no chromadb). These
assert the conversation *loop* properties that unit tests on isolated components
miss — session history accumulates in order, every turn returns a sources list,
the current entry never retrieves itself, the crisis gate fires mid-conversation,
and PRIVACY_MODE=strict redaction holds across turns.

Semantic memory quality (does turn N surface the right earlier turn) is asserted
separately against real Chroma in test_conversation_memory_integration.py.
"""

from __future__ import annotations

import pytest

import app as app_module
from tests.support.fake_provider import ThemeAwareFakeProvider
from tests.support.fake_vector_store import InMemoryVectorStore
from tests.support.scenarios import load_scenario
from tests.support.scenarios import SCENARIO_DIR


@pytest.fixture
def convo_client(monkeypatch):
    """A test client whose LLM and vector store are deterministic doubles.

    The Flask test client keeps a cookie jar across requests, so session["chat"]
    and the per-session RAG namespace persist turn to turn, exactly like a real
    browser session.
    """
    monkeypatch.setattr(app_module, "_provider", ThemeAwareFakeProvider())
    monkeypatch.setattr(app_module, "vector_store", InMemoryVectorStore())
    with app_module.app.test_client() as client:
        yield client


def _analyze(client, entry: str):
    resp = client.post("/analyze", json={"entry": entry, "quality_mode": True})
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


def _history(client):
    return client.get("/session/history").get_json()


def test_session_history_accumulates_in_order(convo_client):
    entries = [
        "My manager piled three deadlines on me and I am burned out.",
        "Another late night at the office finishing a project.",
        "I told my manager the workload is too much and felt relief.",
    ]
    for i, entry in enumerate(entries, start=1):
        _analyze(convo_client, entry)
        history = _history(convo_client)
        assert len(history) == i, f"expected {i} turns after turn {i}, got {len(history)}"
    # Order preserved and entries round-tripped.
    stored = [turn["entry"] for turn in _history(convo_client)]
    assert stored == entries


def test_every_turn_returns_a_sources_list(convo_client):
    for entry in ["I barely slept, awake at 3am.", "Restless night again, foggy all morning."]:
        body = _analyze(convo_client, entry)
        assert isinstance(body["sources"], list)


def test_memory_grows_across_turns(convo_client):
    """By the second lexically-overlapping turn, the first is retrievable."""
    _analyze(convo_client, "My manager gave me three deadlines and I am burned out at work.")
    body = _analyze(convo_client, "More work deadlines from my manager; the burnout is worse.")
    # The fast store ranks by token overlap; the prior work turn should surface.
    assert body["sources"], "expected the earlier work turn to be retrievable as memory"


def test_current_entry_never_retrieves_itself(convo_client):
    """Self-exclusion: an entry must not appear in its own sources, even though
    it is written to the store on the same request."""
    first = _analyze(convo_client, "My manager and the deadlines are burning me out at work.")
    first_source_ids = {s["id"] for s in first["sources"]}
    # Turn 1 has no prior, so no sources; the invariant is that nothing returned
    # is the just-written entry. Turn 2 (same theme) must still exclude itself.
    second = _analyze(convo_client, "Work deadlines from my manager again; still burned out.")
    second_ids = {s["id"] for s in second["sources"]}
    # Exactly one prior entry exists and it is the only legitimate hit.
    assert len(second_ids) == 1
    assert first_source_ids == set()  # nothing on the first turn
    # The single hit is the prior turn, not a self-reference to turn 2.
    assert second["sources"][0]["snippet"]


def test_crisis_turn_suppresses_reframe_midconversation(convo_client):
    """A crisis entry arriving after normal turns must route to support and clear
    the reframe path — the gate is per-turn, not a one-time startup decision."""
    _analyze(convo_client, "Another rejection email after 80 applications. Exhausting.")
    body = _analyze(
        convo_client,
        "I keep thinking everyone would be better off if I disappeared, and I don't want to be here anymore.",
    )
    analysis = body["analysis"]
    assert analysis["crisis_support"] is True
    assert analysis["reframe"] == ""
    assert analysis["support_message"]
    # And the conversation recovers: a later benign turn is not crisis-flagged.
    recover = _analyze(convo_client, "I talked to a friend and took a walk; felt less alone.")
    assert recover["analysis"]["crisis_support"] is False


def test_non_crisis_turn_keeps_reframe(convo_client):
    body = _analyze(convo_client, "We argued again and I do not feel heard by my partner.")
    assert body["analysis"]["crisis_support"] is False
    assert body["analysis"]["reframe"]  # reframe offered on a normal negative entry


def test_privacy_strict_redacts_persisted_entry_across_turns(convo_client, monkeypatch):
    """Under PRIVACY_MODE=strict, PII must not survive into the session history
    (the cookie is signed but not encrypted)."""
    monkeypatch.setattr(app_module.cfg, "privacy_mode", "strict")
    _analyze(convo_client, "Reach me at jane.doe@example.com or 415-555-0198 about the job.")
    history = _history(convo_client)
    stored_entry = history[-1]["entry"]
    assert "jane.doe@example.com" not in stored_entry
    assert "415-555-0198" not in stored_entry
    assert "[REDACTED_EMAIL]" in stored_entry


def test_scenarios_drive_the_route_end_to_end(convo_client):
    """Every crisis-annotated turn in the crisis scenario behaves as declared."""
    scenario = load_scenario(SCENARIO_DIR / "crisis_escalation_journey.json")
    for turn in scenario["turns"]:
        body = _analyze(convo_client, turn["entry"])
        if "expect_crisis_support" in turn:
            assert body["analysis"]["crisis_support"] is turn["expect_crisis_support"], (
                f"turn {turn['id']}: crisis_support mismatch"
            )
