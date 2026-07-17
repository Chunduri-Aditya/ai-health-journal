"""Tests for PRIVACY_MODE=strict redaction at the RAG ingestion boundary.

The guarantee under audit: when strict, no email/phone from a journal entry is
persisted to the local vector store (document text or metadata). Balanced keeps
raw text. These run with no LLM and no real Chroma — they intercept add_entry.
"""

import app


class _CaptureStore:
    """Stands in for the module-global vector_store; records add_entry calls."""

    enabled = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add_entry(self, entry_id=None, text="", metadata=None, namespace=None) -> None:
        self.calls.append({"text": text, "metadata": metadata or {}})


def test_strict_mode_redacts_entry_and_metadata(monkeypatch):
    store = _CaptureStore()
    monkeypatch.setattr(app, "vector_store", store)
    monkeypatch.setattr(app.cfg, "privacy_mode", "strict")

    app._store_in_rag(
        "reach me at jane@example.com or 555-123-4567, I felt anxious",
        "user shared contact jane@example.com",
        entry_id="e1",
        namespace="ns",
    )

    stored = store.calls[0]["text"]
    preview = store.calls[0]["metadata"]["insight_preview"]
    assert "jane@example.com" not in stored
    assert "555-123-4567" not in stored
    assert "[REDACTED_EMAIL]" in stored and "[REDACTED_PHONE]" in stored
    # Metadata must not leak PII either.
    assert "jane@example.com" not in preview


def test_balanced_mode_stores_raw(monkeypatch):
    store = _CaptureStore()
    monkeypatch.setattr(app, "vector_store", store)
    monkeypatch.setattr(app.cfg, "privacy_mode", "balanced")

    app._store_in_rag("email jane@example.com", "insight", entry_id="e2", namespace="ns")

    assert "jane@example.com" in store.calls[0]["text"]


def test_maybe_redact_is_noop_when_not_strict(monkeypatch):
    monkeypatch.setattr(app.cfg, "privacy_mode", "balanced")
    text = "call 555-123-4567"
    assert app._maybe_redact(text) == text
