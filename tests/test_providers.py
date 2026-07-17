"""
Provider factory gate tests.

Tests for the LLM_BACKEND / ALLOW_CLOUD_LLM gate logic.  No Ollama calls,
no Anthropic SDK calls, no network.  The critical invariant:

    LLM_BACKEND=anthropic + ALLOW_CLOUD_LLM=false  →  OllamaProvider
    LLM_BACKEND=anthropic + ALLOW_CLOUD_LLM=true   →  AnthropicProvider
                                                       (only when key present)
    LLM_BACKEND=ollama (default)                   →  OllamaProvider always
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import llm_client
from config import load_config
from providers.factory import get_llm_provider
from providers.ollama_provider import OllamaProvider
from schemas.analysis import AnalysisOutput


def _cfg(**overrides):
    """Build a Config with sensible defaults, overriding specific fields."""
    from dataclasses import asdict, replace
    base = load_config()
    for k, v in overrides.items():
        base = replace(base, **{k: v})
    return base


class TestProviderGate:
    def test_default_backend_is_ollama(self):
        """No env changes: OllamaProvider is always returned."""
        cfg = _cfg(llm_backend="ollama", allow_cloud_llm=False)
        provider = get_llm_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_anthropic_blocked_when_gate_closed(self):
        """
        LLM_BACKEND=anthropic but ALLOW_CLOUD_LLM=false → OllamaProvider.

        The anthropic.Anthropic() client must NOT be instantiated; the API key
        must NOT be read into memory.  We verify this by ensuring no
        AnthropicProvider is returned even when the key is present in the env.
        """
        cfg = _cfg(llm_backend="anthropic", allow_cloud_llm=False)

        # Even with a key in env, the gate must block cloud instantiation.
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-fake-key-for-gate-test"}):
            provider = get_llm_provider(cfg)

        assert isinstance(provider, OllamaProvider), (
            "Gate must return OllamaProvider when ALLOW_CLOUD_LLM=false, "
            "regardless of whether ANTHROPIC_API_KEY is set."
        )

    def test_anthropic_blocked_when_key_absent(self):
        """
        LLM_BACKEND=anthropic + ALLOW_CLOUD_LLM=true but no key → OllamaProvider.
        """
        cfg = _cfg(llm_backend="anthropic", allow_cloud_llm=True)
        env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

        with patch.dict(os.environ, env_without_key, clear=True):
            provider = get_llm_provider(cfg)

        assert isinstance(provider, OllamaProvider), (
            "Gate must fall back to OllamaProvider when ANTHROPIC_API_KEY is absent."
        )

    def test_anthropic_provider_returned_when_gate_open(self):
        """
        LLM_BACKEND=anthropic + ALLOW_CLOUD_LLM=true + key present
        → AnthropicProvider.

        AnthropicProvider is imported lazily inside get_llm_provider, so we
        patch it at its definition site rather than the factory's local name.
        """
        from providers.anthropic_provider import AnthropicProvider

        cfg = _cfg(llm_backend="anthropic", allow_cloud_llm=True)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-fake-key-open-gate"}):
            # Patch at the module level where AnthropicProvider is defined so
            # the lazy import inside factory picks up the patched class.
            with patch("providers.anthropic_provider.AnthropicProvider", AnthropicProvider):
                try:
                    provider = get_llm_provider(cfg)
                    assert isinstance(provider, AnthropicProvider), (
                        "Gate open + key present should return AnthropicProvider."
                    )
                except ImportError:
                    pytest.skip("anthropic SDK not installed; gate open path not testable")

    def test_unknown_backend_defaults_to_ollama(self):
        """Any backend string other than 'anthropic' → OllamaProvider."""
        cfg = _cfg(llm_backend="gpt4all", allow_cloud_llm=True)
        provider = get_llm_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_anthropic_client_never_instantiated_when_gate_closed(self):
        """
        When the gate is closed, AnthropicProvider is never instantiated.
        Patch at the definition site and assert it is never called.
        """
        from unittest.mock import MagicMock
        mock_class = MagicMock()

        cfg = _cfg(llm_backend="anthropic", allow_cloud_llm=False)

        with patch("providers.anthropic_provider.AnthropicProvider", mock_class):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-fake"}):
                provider = get_llm_provider(cfg)

        mock_class.assert_not_called(), (
            "AnthropicProvider must not be instantiated when ALLOW_CLOUD_LLM=false."
        )
        assert isinstance(provider, OllamaProvider)


class TestOllamaProviderDelegation:
    """OllamaProvider is a behavior-preserving wrapper — existing tests cover the
    actual logic; here we just verify delegation happens without errors."""

    def test_healthcheck_delegates_to_llm_client(self, monkeypatch):
        import providers.ollama_provider as op_mod
        import llm_client

        monkeypatch.setattr(llm_client, "check_ollama_available", lambda: True)
        provider = OllamaProvider()
        assert provider.healthcheck() is True

    def test_generate_delegates_to_ollama_generate(self, monkeypatch):
        import llm_client

        monkeypatch.setattr(llm_client, "ollama_generate", lambda *a, **kw: "hello")
        provider = OllamaProvider()
        result = provider.generate("model", "prompt")
        assert result == "hello"

    def test_json_generate_delegates_and_returns_dict(self, monkeypatch):
        fake = {"summary": "ok", "emotions": [], "patterns": [], "triggers": [],
                "coping_suggestions": [], "quotes_from_user": [], "confidence": 0.5}
        monkeypatch.setattr(llm_client, "ollama_generate", lambda *a, **kw: json.dumps(fake))
        provider = OllamaProvider()
        result = provider.json_generate("model", "sys", "user")
        assert result == fake


class TestOllamaProviderValidatorPassthrough:
    """validator_model is forwarded through OllamaProvider to llm_client.json_generate."""

    def test_json_generate_passes_validator_model(self, monkeypatch, valid_analysis_json):
        """
        validator_model=AnalysisOutput travels from OllamaProvider.json_generate
        through llm_client.json_generate, which validates the parsed dict and
        returns a model-dump that satisfies AnalysisOutput.model_validate.
        """
        monkeypatch.setattr(
            llm_client, "ollama_generate", lambda *a, **kw: json.dumps(valid_analysis_json)
        )
        provider = OllamaProvider()
        result = provider.json_generate(
            "phi3:3.8b",
            "system prompt",
            "user prompt",
            validator_model=AnalysisOutput,
        )
        AnalysisOutput.model_validate(result)
        assert result["summary"] == valid_analysis_json["summary"]


class TestAnthropicProviderJsonGenerate:
    """AnthropicProvider.json_generate — mocked SDK, zero network calls."""

    def _make_client(self, tool_input):
        """Return a mock Anthropic client whose messages.create returns a tool_use block."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = tool_input
        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        return mock_client

    def test_valid_payload_passes_pydantic_validation(self, valid_analysis_json):
        """
        A tool_use block with a valid AnalysisOutput payload passes validation
        and json_generate returns a dict that satisfies AnalysisOutput.model_validate.
        """
        from providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        with patch.object(provider, "_client", return_value=self._make_client(valid_analysis_json)):
            result = provider.json_generate(
                "claude-sonnet-4-5",
                "system prompt",
                "user prompt",
                validator_model=AnalysisOutput,
            )
        AnalysisOutput.model_validate(result)
        assert result["summary"] == valid_analysis_json["summary"]

    def test_empty_dict_raises_validation_error(self):
        """
        A tool_use block with {} fails AnalysisOutput validation and raises
        ValueError("json_schema_validation_failed").
        """
        from providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        with patch.object(provider, "_client", return_value=self._make_client({})):
            with pytest.raises(ValueError, match="json_schema_validation_failed"):
                provider.json_generate(
                    "claude-sonnet-4-5",
                    "system prompt",
                    "user prompt",
                    validator_model=AnalysisOutput,
                )


class TestAnthropicProviderGenerate:
    """AnthropicProvider.generate — mocked SDK, zero network calls."""

    def test_generate_returns_content_text_stripped(self):
        """response.content[0].text is returned with leading/trailing whitespace removed."""
        from providers.anthropic_provider import AnthropicProvider

        content_block = MagicMock()
        content_block.text = "  hello world  "
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="test-key")
        with patch.object(provider, "_client", return_value=mock_client):
            result = provider.generate("claude-sonnet-4-5", "test prompt")

        assert result == "hello world"
