"""Tests for the OpenStax Psychology 2e reference corpus wiring.

Covers the retrieval gate, the distinct citation formatting, and the route-level
response shape. Does not cover whether a live LLM actually honors the
attribution/no-diagnosis prompt rules (generator_prompts.py rule 13,
verifier_prompts.py rule 9) -- that is a live-model behavior, not something
mechanically testable offline. What IS tested here, and testable by
construction rather than by trusting a prompt instruction: the deterministic
crisis floor (_is_crisis) never receives reference_context as a parameter at
all, so it cannot be influenced by reference material regardless of what any
prompt says.
"""

from __future__ import annotations

import inspect

import pytest

import app as app_module
from tests.support.fake_vector_store import InMemoryVectorStore


@pytest.fixture
def flask_client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client


@pytest.fixture
def healthy_provider(monkeypatch):
    monkeypatch.setattr(app_module._provider, "healthcheck", lambda: True)


@pytest.fixture
def fake_store(monkeypatch):
    store = InMemoryVectorStore()
    monkeypatch.setattr(app_module, "vector_store", store)
    return store


def _seed_reference_passage(store: InMemoryVectorStore, namespace: str) -> None:
    store.add_entry(
        entry_id="ref_14_2_0",
        text="Job strain and heavy workload are among the greatest risk factors for burnout.",
        metadata={
            "kind": "reference_passage",
            "source_book": "OpenStax Psychology 2e",
            "license": "CC BY-NC-SA 4.0",
            "chapter": 14,
            "section": "14.2",
            "section_title": "Stressors",
            "source_url": "https://openstax.org/books/psychology-2e/pages/14-2-stressors",
            "attribution": "Access for free at https://openstax.org/books/psychology-2e/pages/14-2-stressors",
        },
        namespace=namespace,
    )


class TestRetrieveReferenceHits:
    def test_disabled_by_default_returns_nothing(self, fake_store, monkeypatch):
        """REFERENCE_CORPUS_ENABLED defaults false; must not silently retrieve."""
        _seed_reference_passage(fake_store, app_module.cfg.reference_namespace)
        monkeypatch.setattr(app_module.cfg, "reference_corpus_enabled", False)
        assert app_module._retrieve_reference_hits("burned out from workload") == []

    def test_enabled_with_data_returns_hits(self, fake_store, monkeypatch):
        _seed_reference_passage(fake_store, app_module.cfg.reference_namespace)
        monkeypatch.setattr(app_module.cfg, "reference_corpus_enabled", True)
        hits = app_module._retrieve_reference_hits("burned out from workload")
        assert len(hits) == 1
        assert hits[0].metadata["section_title"] == "Stressors"

    def test_empty_entry_returns_nothing(self, fake_store, monkeypatch):
        _seed_reference_passage(fake_store, app_module.cfg.reference_namespace)
        monkeypatch.setattr(app_module.cfg, "reference_corpus_enabled", True)
        assert app_module._retrieve_reference_hits("") == []


class TestFormatting:
    def test_reference_context_is_labeled_distinctly_from_journal_context(self, fake_store, monkeypatch):
        """Must not read as one more of the user's own past entries."""
        _seed_reference_passage(fake_store, app_module.cfg.reference_namespace)
        monkeypatch.setattr(app_module.cfg, "reference_corpus_enabled", True)
        hits = app_module._retrieve_reference_hits("burned out from workload")
        formatted = app_module._format_reference_context(hits)
        assert "Reference" in formatted
        assert "OpenStax Psychology 2e" in formatted
        assert "Stressors" in formatted
        # The journal-history formatter's header shape must not appear here --
        # if it did, the two evidence pools would be visually indistinguishable
        # to the model reading the prompt.
        assert "Retrieved Context" not in formatted

    def test_empty_hits_produce_empty_string(self):
        assert app_module._format_reference_context([]) == ""

    def test_source_serialization_carries_attribution(self, fake_store, monkeypatch):
        _seed_reference_passage(fake_store, app_module.cfg.reference_namespace)
        monkeypatch.setattr(app_module.cfg, "reference_corpus_enabled", True)
        hits = app_module._retrieve_reference_hits("burned out from workload")
        source = app_module._reference_hit_to_source(hits[0])
        assert source["attribution"] == "Access for free at https://openstax.org/books/psychology-2e/pages/14-2-stressors"
        assert source["license"] == "CC BY-NC-SA 4.0"
        assert source["url"].startswith("https://openstax.org/")


class TestCrisisFloorIsStructurallyIsolated:
    """The deterministic crisis floor cannot be influenced by reference material,
    not because a prompt says not to, but because the function that implements
    it never receives reference_context as an argument at all."""

    def test_is_crisis_has_no_reference_context_parameter(self):
        sig = inspect.signature(app_module._is_crisis)
        assert "reference_context" not in sig.parameters

    def test_apply_reframe_gate_has_no_reference_context_parameter(self):
        sig = inspect.signature(app_module._apply_reframe_gate)
        assert "reference_context" not in sig.parameters


class TestRouteLevel:
    def test_analyze_response_includes_reference_sources_field(
        self, flask_client, healthy_provider, fake_store, monkeypatch
    ):
        """Even with the corpus disabled (the default), the response shape is
        stable: reference_sources is always present, just empty, so a client
        does not need to branch on whether the feature is turned on."""
        monkeypatch.setattr(app_module.cfg, "quality_mode_default", True)

        def fake_json_generate(model, system, user, **kwargs):
            validator = kwargs.get("validator_model")
            if validator and validator.__name__ == "VerifierVerdict":
                return {
                    "groundedness_score": 0.95,
                    "unsupported_claims": [],
                    "safety_flags": [],
                    "crisis_detected": False,
                    "rewrite_required": False,
                    "rewrite_instructions": "",
                }
            return {
                "summary": "Feeling stressed about work.",
                "emotions": ["stress"],
                "patterns": [],
                "triggers": [],
                "coping_suggestions": ["Take a short break."],
                "quotes_from_user": [],
                "confidence": 0.8,
            }

        monkeypatch.setattr(app_module._provider, "json_generate", fake_json_generate)

        resp = flask_client.post(
            "/analyze", json={"entry": "Long week at work, feeling stretched thin.", "quality_mode": True}
        )
        body = resp.get_json()
        assert resp.status_code == 200
        assert "reference_sources" in body
        assert body["reference_sources"] == []
