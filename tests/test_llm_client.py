"""
Tests for llm_client.json_generate and its helpers.

No Ollama calls happen here; we monkeypatch ollama_generate with canned
responses and assert parsing, retry, and validation behavior.
"""

from __future__ import annotations

import json
from typing import Callable, List

import pytest

import llm_client
from llm_client import (
    _parse_json_lenient,
    _strip_markdown_fences,
    extract_json_substring,
    json_generate,
)
from schemas.analysis import AnalysisOutput
from schemas.verifier import VerifierVerdict


def _queue_responses(monkeypatch, responses: List[str]) -> List[str]:
    """
    Replace ollama_generate with a queue that pops one response per call.
    Returns a list of the prompts received, so tests can assert on retry
    strictness escalation.
    """
    seen_prompts: List[str] = []
    remaining = list(responses)

    def fake_generate(model, prompt, **_kwargs):
        seen_prompts.append(prompt)
        if not remaining:
            raise AssertionError("ollama_generate called more times than expected")
        return remaining.pop(0)

    monkeypatch.setattr(llm_client, "ollama_generate", fake_generate)
    return seen_prompts


class TestStripMarkdownFences:
    def test_no_fences_passthrough(self):
        assert _strip_markdown_fences("{\"a\": 1}") == "{\"a\": 1}"

    def test_json_fence(self):
        raw = "```json\n{\"a\": 1}\n```"
        assert _strip_markdown_fences(raw) == "{\"a\": 1}"

    def test_plain_fence(self):
        raw = "```\n{\"a\": 1}\n```"
        assert _strip_markdown_fences(raw) == "{\"a\": 1}"


class TestExtractJsonSubstring:
    def test_simple_object(self):
        assert extract_json_substring('noise {"a": 1} tail') == '{"a": 1}'

    def test_no_braces_raises(self):
        with pytest.raises(ValueError):
            extract_json_substring("no json here")

    def test_only_open_brace_raises(self):
        with pytest.raises(ValueError):
            extract_json_substring("{ incomplete")


class TestParseJsonLenient:
    def test_direct_parse(self):
        assert _parse_json_lenient('{"a": 1}') == {"a": 1}

    def test_with_fence(self):
        assert _parse_json_lenient('```json\n{"a": 1}\n```') == {"a": 1}

    def test_with_preamble(self):
        assert _parse_json_lenient('Here you go: {"a": 1} enjoy') == {"a": 1}


class TestJsonGenerateBasic:
    def test_happy_path_dict_return(self, monkeypatch, valid_analysis_json):
        _queue_responses(monkeypatch, [json.dumps(valid_analysis_json)])
        out = json_generate("model", "sys", "user", max_retries=1)
        assert out == valid_analysis_json

    def test_retries_on_garbage_then_succeeds(self, monkeypatch, valid_analysis_json):
        _queue_responses(
            monkeypatch,
            ["<<not json at all>>", json.dumps(valid_analysis_json)],
        )
        out = json_generate("model", "sys", "user", max_retries=3)
        assert out["summary"] == valid_analysis_json["summary"]

    def test_all_retries_fail_raises_parse_error(self, monkeypatch):
        _queue_responses(monkeypatch, ["not json", "still not json"])
        with pytest.raises(ValueError, match="json_parse_failed"):
            json_generate("model", "sys", "user", max_retries=2)

    def test_strict_reminder_added_from_attempt_three(self, monkeypatch):
        """
        On attempt index >= 2, the user prompt should gain a REMINDER suffix.
        """
        seen = _queue_responses(
            monkeypatch,
            ["oops", "still bad", '{"summary":"hi","emotions":[],"patterns":[],"triggers":[],"coping_suggestions":[],"quotes_from_user":[],"confidence":0.5}'],
        )
        json_generate("model", "sys", "user prompt", max_retries=3)
        assert len(seen) == 3
        assert "REMINDER" not in seen[0]
        assert "REMINDER" not in seen[1]
        assert "REMINDER" in seen[2]


class TestJsonGenerateWithValidator:
    def test_empty_object_is_rejected_and_retried(
        self, monkeypatch, valid_analysis_json
    ):
        """
        The DPO-critical gap: a model returning {} should NOT silently pass.
        Validator should force a retry; retry succeeds on the second response.
        """
        _queue_responses(
            monkeypatch,
            ["{}", json.dumps(valid_analysis_json)],
        )
        out = json_generate(
            "model",
            "sys",
            "user",
            max_retries=3,
            validator_model=AnalysisOutput,
        )
        assert isinstance(out, dict)
        assert out["summary"] == valid_analysis_json["summary"]

    def test_empty_object_all_retries_exhausted_raises(self, monkeypatch):
        _queue_responses(monkeypatch, ["{}", "{}", "{}"])
        with pytest.raises(ValueError, match="json_schema_validation_failed"):
            json_generate(
                "model",
                "sys",
                "user",
                max_retries=3,
                validator_model=AnalysisOutput,
            )

    def test_null_summary_rejected_until_valid(
        self, monkeypatch, valid_analysis_json
    ):
        payload = dict(valid_analysis_json)
        payload["summary"] = ""  # fails min_length=1
        _queue_responses(
            monkeypatch,
            [json.dumps(payload), json.dumps(valid_analysis_json)],
        )
        out = json_generate(
            "model",
            "sys",
            "user",
            max_retries=2,
            validator_model=AnalysisOutput,
        )
        assert out["summary"] == valid_analysis_json["summary"]

    def test_return_model_returns_pydantic_instance(
        self, monkeypatch, valid_analysis_json
    ):
        _queue_responses(monkeypatch, [json.dumps(valid_analysis_json)])
        out = json_generate(
            "model",
            "sys",
            "user",
            max_retries=1,
            validator_model=AnalysisOutput,
            return_model=True,
        )
        assert isinstance(out, AnalysisOutput)
        assert out.summary == valid_analysis_json["summary"]

    def test_verifier_verdict_path(self, monkeypatch, valid_verifier_json):
        _queue_responses(monkeypatch, [json.dumps(valid_verifier_json)])
        out = json_generate(
            "model",
            "sys",
            "user",
            max_retries=1,
            validator_model=VerifierVerdict,
        )
        assert out["rewrite_required"] is False
        assert out["groundedness_score"] == pytest.approx(0.88)

    def test_backwards_compatible_without_validator(
        self, monkeypatch, valid_analysis_json
    ):
        """Leaving validator_model=None keeps today's dict-shaped behavior."""
        _queue_responses(monkeypatch, [json.dumps(valid_analysis_json)])
        out = json_generate("model", "sys", "user", max_retries=1)
        assert isinstance(out, dict)
        assert out == valid_analysis_json
