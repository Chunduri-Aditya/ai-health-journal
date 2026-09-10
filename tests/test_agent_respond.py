from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

from src.agent.respond import (
    AGENT_RESPOND_SYSTEM_PROMPT,
    build_respond_user_prompt,
    generate_grounded_response,
)


class FakeProvider:
    def __init__(self, *, draft: str, verdict: Optional[Dict[str, Any]] = None, revised: str = "") -> None:
        self.draft = draft
        self.verdict = verdict or {}
        self.revised = revised
        self.generate_calls = []
        self.json_calls = []

    def generate(self, model, prompt, *, system=None, temperature=None, timeout=30):
        self.generate_calls.append({"model": model, "prompt": prompt, "system": system})
        # First generate is draft; later is revise
        if len([c for c in self.generate_calls if c["system"] == AGENT_RESPOND_SYSTEM_PROMPT]) == 1:
            return self.draft
        return self.revised or self.draft

    def json_generate(self, model, system_prompt, user_prompt, **kwargs):
        self.json_calls.append({"model": model, "system": system_prompt, "user": user_prompt})
        return dict(self.verdict)

    def healthcheck(self) -> bool:
        return True


def _cfg(**overrides):
    base = dict(
        llm_backend="ollama",
        allow_cloud_llm=False,
        generator_model="gen",
        verifier_model="ver",
        fallback_model="fall",
        anthropic_generator_model="claude-gen",
        anthropic_verifier_model="claude-ver",
        quality_mode_default=True,
        groundedness_threshold=0.75,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestRespondPrompts:
    def test_prompt_includes_context(self):
        p = build_respond_user_prompt("how was work?", "I argued at work")
        assert "USER_MESSAGE" in p
        assert "I argued at work" in p
        assert "RETRIEVED_CONTEXT" in p

    def test_prompt_marks_empty_context(self):
        p = build_respond_user_prompt("how was work?", "")
        assert "RETRIEVED_CONTEXT: (none)" in p
        assert "Do not invent history" in p


class TestGenerateGroundedResponse:
    def test_passes_when_grounded(self):
        provider = FakeProvider(
            draft="You mentioned arguing at work.",
            verdict={
                "groundedness_score": 0.95,
                "unsupported_claims": [],
                "safety_flags": [],
                "rewrite_required": False,
                "rewrite_instructions": "",
            },
        )
        out = generate_grounded_response(
            provider,
            _cfg(),
            user_message="how was work?",
            retrieved_context="I argued at work",
        )
        assert out["answer"] == "You mentioned arguing at work."
        assert out["verified"] is True
        assert out["revised"] is False
        assert len(provider.json_calls) == 1

    def test_revises_when_ungrounded(self):
        provider = FakeProvider(
            draft="You got promoted last year.",
            revised="I only see an argument at work in your retrieved entries.",
            verdict={
                "groundedness_score": 0.2,
                "unsupported_claims": ["got promoted last year"],
                "safety_flags": [],
                "rewrite_required": True,
                "rewrite_instructions": "Remove invented promotion.",
            },
        )
        out = generate_grounded_response(
            provider,
            _cfg(),
            user_message="how was work?",
            retrieved_context="I argued at work",
        )
        assert out["revised"] is True
        assert out["verified"] is True
        assert "promotion" not in out["answer"].lower()
        assert "argument" in out["answer"].lower()

    def test_skips_verify_when_quality_mode_off(self):
        provider = FakeProvider(draft="plain answer")
        out = generate_grounded_response(
            provider,
            _cfg(quality_mode_default=False),
            user_message="hello",
            retrieved_context="",
        )
        assert out["answer"] == "plain answer"
        assert out["verified"] is False
        assert provider.json_calls == []

    def test_uses_anthropic_models_when_gated_on(self):
        provider = FakeProvider(
            draft="ok",
            verdict={
                "groundedness_score": 1.0,
                "unsupported_claims": [],
                "safety_flags": [],
                "rewrite_required": False,
                "rewrite_instructions": "",
            },
        )
        out = generate_grounded_response(
            provider,
            _cfg(llm_backend="anthropic", allow_cloud_llm=True),
            user_message="hello",
        )
        assert provider.generate_calls[0]["model"] == "claude-gen"
        assert provider.json_calls[0]["model"] == "claude-ver"
        assert out["model"] == "claude-gen"
