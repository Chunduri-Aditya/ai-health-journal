# GateGuard: callers pytest. Affected API: health + production API-key gate.
# Data schemas: none. User: Implement the plan as specified. Do NOT edit the plan file.
from __future__ import annotations

import os

import pytest

# Force noop store before importing the app module.
os.environ.setdefault("ENV", "test")
os.environ["RETRIEVAL_ENABLED"] = "false"
os.environ["VECTOR_BACKEND"] = "none"
os.environ["LLM_BACKEND"] = "ollama"
os.environ["ALLOW_CLOUD_LLM"] = "false"


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from service.main import app

    return TestClient(app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "retrieval" in body


def test_readyz(client):
    r = client.get("/readyz")
    assert r.status_code == 200


def test_root_landing(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Journal Agent" in r.text
    assert "/docs" in r.text
    assert "/v1/agent/invoke" in r.text


def test_api_key_required_outside_dev(monkeypatch):
    """ENV=production without JOURNAL_AGENT_API_KEY → 503 on /v1 routes."""
    import service.main as svc
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("JOURNAL_AGENT_API_KEY", raising=False)
    client = TestClient(svc.app)
    r = client.post(
        "/v1/agent/invoke",
        json={"message": "hello", "session_id": "auth-test"},
    )
    assert r.status_code == 503
    assert "JOURNAL_AGENT_API_KEY" in r.json()["detail"]


def test_api_key_rejects_wrong_key(monkeypatch):
    import service.main as svc
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("JOURNAL_AGENT_API_KEY", "correct-secret")
    client = TestClient(svc.app)
    r = client.post(
        "/v1/agent/invoke",
        json={"message": "hello", "session_id": "auth-test"},
        headers={"X-API-Key": "wrong"},
    )
    assert r.status_code == 401
