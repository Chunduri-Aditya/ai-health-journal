"""Tests for the chat-history -> DPO training-data pipeline.

`evals/build_dpo_dataset.py` decides what becomes preference-training data, yet
had no coverage. These lock in the gates that matter for a *health* app: forbidden
content (diagnosis / dosage / data-exfiltration language) is caught, and only
genuinely-better, fully-grounded quality outputs are kept as "chosen". Also
guards the shape of chat_history.json as a training-pair source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.build_dpo_dataset import (
    build_prompt,
    contains_forbidden_content,
    should_keep_pair,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── contains_forbidden_content ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "text",
    [
        "You clearly have a diagnosis of major depression.",
        "You should take 50 mg of the medication every morning.",
        "I will upload to cloud everything you write here.",
        "I am 100% certain you have an anxiety disorder.",
    ],
)
def test_forbidden_content_is_caught(text):
    assert contains_forbidden_content(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "It sounds like today was heavy. Naming that is a real step.",
        "You might consider talking to someone you trust.",
        "That frustration makes sense given the deadlines you described.",
    ],
)
def test_benign_content_passes(text):
    assert contains_forbidden_content(text) is False


# ── should_keep_pair ───────────────────────────────────────────────────────
def _strong_quality():
    return {
        "parse_failures": 0,
        "metrics": {
            "faithfulness": 1.0,
            "no_invention": 1.0,
            "answer_relevancy": 0.9,
            "unsupported_claims": [],
        },
        "quality_output": '{"summary": "grounded and safe"}',
    }


def test_pair_rejected_on_quality_parse_failure():
    quality = _strong_quality()
    quality["parse_failures"] = 1
    keep, reason = should_keep_pair({"metrics": {}}, quality)
    assert keep is False
    assert reason == "quality_parse_failure"


def test_pair_rejected_when_quality_faithfulness_too_low():
    quality = _strong_quality()
    quality["metrics"]["faithfulness"] = 0.90
    keep, reason = should_keep_pair({"metrics": {}, "baseline_output": "x"}, quality)
    assert keep is False
    assert reason == "quality_faithfulness_too_low"


def test_pair_rejected_when_quality_invents():
    quality = _strong_quality()
    quality["metrics"]["no_invention"] = 0.90
    keep, reason = should_keep_pair({"metrics": {}, "baseline_output": "x"}, quality)
    assert keep is False
    assert reason == "quality_no_invention_not_perfect"


def test_legacy_baseline_is_always_kept():
    """A missing/empty baseline output means baseline was legacy free-text, so a
    strong quality output is preferred by construction."""
    keep, reason = should_keep_pair({"metrics": {}, "baseline_output": ""}, _strong_quality())
    assert keep is True
    assert reason == "baseline_legacy_format"


def test_forbidden_baseline_makes_quality_the_winner():
    baseline = {
        "metrics": {"faithfulness": 0.99, "no_invention": 1.0, "answer_relevancy": 0.99},
        "baseline_output": "You have a diagnosis of depression and should take 50 mg daily.",
    }
    keep, reason = should_keep_pair(baseline, _strong_quality())
    assert keep is True
    assert reason == "baseline_contains_forbidden_content"


def test_quality_kept_when_strictly_more_faithful():
    baseline = {
        "metrics": {"faithfulness": 0.80, "no_invention": 1.0, "answer_relevancy": 0.9, "unsupported_claims": []},
        "baseline_output": "a plausible but weaker analysis",
    }
    keep, reason = should_keep_pair(baseline, _strong_quality())
    assert keep is True
    assert reason == "quality_better"


# ── build_prompt ───────────────────────────────────────────────────────────
def test_build_prompt_omits_context_section_when_absent():
    prompt = build_prompt("I feel overwhelmed at work.")
    assert "I feel overwhelmed at work." in prompt
    # The retrieved-context block is headed "RETRIEVED_CONTEXT (from past
    # entries):"; the bare token also appears in the constant rules text, so
    # key off the block header, not the token.
    assert "from past entries" not in prompt


def test_build_prompt_includes_context_when_present():
    prompt = build_prompt("I feel overwhelmed.", retrieved_context="[past] a stressful week")
    assert "from past entries" in prompt
    assert "a stressful week" in prompt


# ── chat_history.json as a training source ─────────────────────────────────
def test_chat_history_has_trainable_shape():
    """chat_history.json feeds preference pairs; every record needs the entry +
    response fields build_prompt / pairing rely on."""
    history = json.loads((REPO_ROOT / "chat_history.json").read_text(encoding="utf-8"))
    assert isinstance(history, list) and history, "chat history should be a non-empty list"
    for record in history:
        assert isinstance(record.get("entry"), str) and record["entry"].strip()
        assert isinstance(record.get("response"), str) and record["response"].strip()
        # Each entry must be usable as a prompt without inventing fields.
        assert record["entry"] in build_prompt(record["entry"])
