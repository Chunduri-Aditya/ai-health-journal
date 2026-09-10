# GateGuard: callers pytest. Affected API: classify_journal_relevance heuristics.
# Data schemas: RelevanceVerdict. User: "simple model to check if the journal
# message is actually relevant... coding questions its kinda stealing".
"""Tests for the journal relevance gate (heuristic path; no network)."""

from __future__ import annotations

from src.journal_gate import OFF_TOPIC_MESSAGE, classify_journal_relevance


def test_blocks_coding_request():
    v = classify_journal_relevance(
        "Write a Python function that sorts a list",
        use_llm=False,
    )
    assert v.relevant is False
    assert v.category == "coding"
    assert v.source == "heuristic"


def test_blocks_code_fence():
    v = classify_journal_relevance(
        "Fix this:\n```python\ndef foo():\n  pass\n```",
        use_llm=False,
    )
    assert v.relevant is False
    assert v.category == "coding"


def test_blocks_short_qa():
    v = classify_journal_relevance(
        "What is the capital of France?",
        use_llm=False,
    )
    assert v.relevant is False
    assert v.category == "qa"


def test_allows_journal_entry():
    v = classify_journal_relevance(
        "Today I felt proud that I paused before answering a hard Slack message.",
        use_llm=False,
    )
    assert v.relevant is True
    assert v.category == "journal"


def test_allows_grateful_prompt_seed():
    v = classify_journal_relevance(
        "Today I'm grateful I noticed that I took a breath before reacting.",
        use_llm=False,
    )
    assert v.relevant is True


def test_off_topic_message_is_warm():
    assert "journaling" in OFF_TOPIC_MESSAGE.lower()
    assert "coding" in OFF_TOPIC_MESSAGE.lower()
