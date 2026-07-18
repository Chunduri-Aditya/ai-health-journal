"""Real-Chroma memory tests: does the conversation remember itself?

Marked `slow` because it loads the ONNX MiniLM embedder and drives full scenarios
through the route. This is what the fast fake-store path in
test_conversation_progression.py cannot prove: that *semantic* retrieval surfaces
the right earlier turn — e.g. a work turn recalls the earlier work turn over the
sleep and relationship turns in between, not merely the most recent entry.

Run: pytest -m slow    (chromadb is a core dependency, so no extra install)
"""

from __future__ import annotations

import tempfile

import pytest

import app as app_module
from tests.support.fake_provider import ThemeAwareFakeProvider
from tests.support.scenarios import (
    SCENARIO_DIR,
    load_scenario,
    resolve_source_ids,
    validate_scenario,
)

pytestmark = pytest.mark.slow


@pytest.fixture
def chroma_convo(monkeypatch):
    """Test client backed by a real, ephemeral Chroma store + the fake provider."""
    from vector_store.chroma_store import ChromaStore

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("CHROMA_PERSIST_DIR", tmp)
        store = ChromaStore(default_namespace="mem_test")
        monkeypatch.setattr(app_module, "vector_store", store)
        monkeypatch.setattr(app_module, "_provider", ThemeAwareFakeProvider())
        with app_module.app.test_client() as client:
            yield client


def _drive(client, scenario):
    """POST every turn in order; return [(turn, resolved_prior_ids, analysis)]."""
    rows = []
    for turn in scenario["turns"]:
        resp = client.post("/analyze", json={"entry": turn["entry"], "quality_mode": True})
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        resolved = resolve_source_ids(body.get("sources", []), scenario, turn["turn"])
        rows.append((turn, resolved, body["analysis"]))
    return rows


def test_mixed_themes_recall_matching_theme_not_recency(chroma_convo):
    """The decisive test: interleaved themes, so the top retrieved memory must be
    the same-theme prior turn, not the most recent one."""
    scenario = load_scenario(SCENARIO_DIR / "mixed_life_journey.json")
    validate_scenario(scenario)
    for turn, resolved, _ in _drive(chroma_convo, scenario):
        expected = turn.get("expect_retrieves_prior_ids")
        if not expected:
            continue
        assert resolved, f"turn {turn['id']} retrieved no prior memory"
        assert resolved[0] in expected, (
            f"turn {turn['id']} top memory was {resolved[0]!r}, expected one of {expected} "
            f"(retrieval followed recency, not theme)"
        )


def test_recurring_theme_resurfaces_across_turns(chroma_convo):
    """A recurring work-burnout arc: every later turn recalls at least one earlier
    work turn, and the top memory is always an expected one."""
    scenario = load_scenario(SCENARIO_DIR / "work_stress_journey.json")
    for turn, resolved, _ in _drive(chroma_convo, scenario):
        expected = turn.get("expect_retrieves_prior_ids")
        if not expected:
            continue
        assert set(expected) & set(resolved), (
            f"turn {turn['id']} recalled none of its expected priors {expected}; got {resolved}"
        )
        assert resolved[0] in expected, (
            f"turn {turn['id']} top memory {resolved[0]!r} not in {expected}"
        )


def test_crisis_gate_fires_midconversation_with_real_memory(chroma_convo):
    """Crisis routing holds even with real retrieved context in the prompt."""
    scenario = load_scenario(SCENARIO_DIR / "crisis_escalation_journey.json")
    for turn, _, analysis in _drive(chroma_convo, scenario):
        if "expect_crisis_support" in turn:
            assert analysis["crisis_support"] is turn["expect_crisis_support"], (
                f"turn {turn['id']}: crisis_support={analysis['crisis_support']}, "
                f"expected {turn['expect_crisis_support']}"
            )
            if turn["expect_crisis_support"]:
                assert analysis["reframe"] == ""
                assert analysis["support_message"]
