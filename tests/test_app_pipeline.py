"""
Integration-style tests confirming defect #12 is wired at the Flask route level.

Defect #12: prior to the fix, a model returning {} would silently pass pydantic
validation and come back as `analysis: {}` in the /analyze response. Fix: pass
validator_model=AnalysisOutput to llm_client.json_generate in _run_quality_pipeline
so {} triggers a retry rather than passing through.

No Ollama server needed. We monkeypatch llm_client.ollama_generate with a
response queue and drive the app through its Flask test client.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

import app as app_module
import llm_client


def _queue_responses(monkeypatch, responses: List[str]) -> None:
    """Replace llm_client.ollama_generate with a queue of canned string responses."""
    remaining = list(responses)

    def fake_generate(model, prompt, **_kwargs):
        if not remaining:
            raise AssertionError("ollama_generate called more times than expected")
        return remaining.pop(0)

    monkeypatch.setattr(llm_client, "ollama_generate", fake_generate)


@pytest.fixture
def flask_client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client


@pytest.fixture
def healthy_provider(monkeypatch):
    """Patch _provider.healthcheck so the /analyze route does not short-circuit."""
    monkeypatch.setattr(app_module._provider, "healthcheck", lambda: True)


class TestDefect12Wired:
    """
    Defect #12: validator_model=AnalysisOutput is passed inside _run_quality_pipeline
    so an empty dict from the LLM is rejected and retried, never reaching the caller.
    """

    def test_empty_draft_is_retried_not_returned(
        self,
        flask_client,
        healthy_provider,
        monkeypatch,
        valid_analysis_json: Dict[str, Any],
        valid_verifier_json: Dict[str, Any],
    ):
        """
        LLM returns {} on the first draft attempt (malformed response).
        validator_model=AnalysisOutput catches it, the retry returns valid JSON,
        and /analyze responds with a properly structured analysis — not {}.
        """
        _queue_responses(
            monkeypatch,
            [
                "{}",                                # draft attempt 1 — fails AnalysisOutput validation
                json.dumps(valid_analysis_json),     # draft attempt 2 — passes
                json.dumps(valid_verifier_json),     # verifier — passes, no revision triggered
            ],
        )

        resp = flask_client.post(
            "/analyze",
            json={"entry": "I felt anxious today.", "quality_mode": True},
        )

        assert resp.status_code == 200, (
            f"Expected 200 but got {resp.status_code}: {resp.data!r}"
        )
        body = resp.get_json()
        analysis = body.get("analysis", {})
        assert analysis != {}, (
            "analysis must not be {} — defect #12 validator must reject empty dicts and retry"
        )
        assert "summary" in analysis, (
            f"Expected 'summary' key in analysis after retry, got: {analysis!r}"
        )
