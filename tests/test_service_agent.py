from __future__ import annotations

import os

import pytest

os.environ.setdefault("ENV", "test")
os.environ["RETRIEVAL_ENABLED"] = "false"
os.environ["VECTOR_BACKEND"] = "none"
os.environ["LLM_BACKEND"] = "ollama"
os.environ["ALLOW_CLOUD_LLM"] = "false"
os.environ.pop("JOURNAL_AGENT_API_KEY", None)


@pytest.fixture
def memory_store():
    from tests.vector_store.fakes import InMemoryVectorStore

    return InMemoryVectorStore(default_namespace="ai-health-journal")


@pytest.fixture
def client_with_memory(memory_store, monkeypatch):
    import src.service.main as svc
    from fastapi.testclient import TestClient

    monkeypatch.setattr(svc, "_store", memory_store)
    # Reset sessions between tests
    with svc._sessions_lock:
        svc._sessions.clear()
    return TestClient(svc.app), memory_store


@pytest.fixture
def client_noop(monkeypatch):
    import src.service.main as svc
    from fastapi.testclient import TestClient
    from src.vector_store.noop_store import NoOpStore

    monkeypatch.setattr(svc, "_store", NoOpStore())
    with svc._sessions_lock:
        svc._sessions.clear()
    return TestClient(svc.app)


def test_save_confirm_persists_clean_text(client_with_memory):
    client, store = client_with_memory
    r1 = client.post(
        "/v1/agent/invoke",
        json={
            "message": "save this: evening walk felt peaceful",
            "session_id": "t-save",
        },
    )
    assert r1.status_code == 200
    body = r1.json()
    assert body["awaiting_confirmation"] is True
    pending = body["tool_trace"][-1]["pending_write"]
    assert pending["text"] == "evening walk felt peaceful"
    assert not pending["text"].startswith(":")

    r2 = client.post(
        "/v1/agent/invoke",
        json={"message": "confirm", "session_id": "t-save"},
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["last_tool"] == "apply_write"
    assert body2["awaiting_confirmation"] is False
    assert "applied" in body2["response"].lower()
    hits = store.query("peaceful", namespace="ai-health-journal", top_k=3)
    assert any("peaceful" in h.text for h in hits)


def test_confirm_on_noop_does_not_claim_applied(client_noop):
    r1 = client_noop.post(
        "/v1/agent/invoke",
        json={
            "message": "save this: should not persist",
            "session_id": "t-noop",
        },
    )
    assert r1.status_code == 200
    assert r1.json()["awaiting_confirmation"] is True

    r2 = client_noop.post(
        "/v1/agent/invoke",
        json={"message": "confirm", "session_id": "t-noop"},
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["last_tool"] == "apply_write"
    apply_events = [e for e in body["tool_trace"] if e.get("tool") == "apply_write"]
    assert apply_events
    assert apply_events[-1]["status"] == "failed"
    assert apply_events[-1].get("reason") == "retrieval_disabled"
    assert "applied" not in body["response"].lower() or "failed" in body["response"].lower()
    assert "disabled" in body["response"].lower()
