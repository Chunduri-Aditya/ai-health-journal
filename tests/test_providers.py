# GateGuard: callers pytest. Affected API: provider gate + openai_compatible tests.
# Data schemas: none. User: Implement the plan as specified, it is attached for
# your reference. Do NOT edit the plan file itself.
"""
Provider factory gate tests.

Tests for the LLM_BACKEND / ALLOW_CLOUD_LLM gate logic.  No Ollama calls,
no Anthropic/OpenAI SDK network calls.  The critical invariant:

    LLM_BACKEND=anthropic + ALLOW_CLOUD_LLM=false  →  OllamaProvider
    LLM_BACKEND=anthropic + ALLOW_CLOUD_LLM=true   →  AnthropicProvider
                                                       (only when key present)
    LLM_BACKEND=openai_compatible + gate closed    →  OllamaProvider
    LLM_BACKEND=openai_compatible + gate + key+url →  OpenAICompatibleProvider
    LLM_BACKEND=ollama (default)                   →  OllamaProvider always
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import src.llm_client as llm_client
from src.config import load_config
from providers.factory import BackendMisconfigured, get_llm_provider
from providers.ollama_provider import OllamaProvider
from providers.roles import resolve_role_models
from schemas.analysis import AnalysisOutput


def _cfg(**overrides):
    """Build a Config with sensible defaults, overriding specific fields."""
    base = load_config()
    if overrides:
        base = base.model_copy(update=overrides)
    return base


class TestProviderGate:
    def test_default_backend_is_ollama(self):
        """No env changes: OllamaProvider is always returned."""
        cfg = _cfg(llm_backend="ollama", allow_cloud_llm=False)
        provider = get_llm_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_anthropic_blocked_when_gate_closed(self):
        cfg = _cfg(llm_backend="anthropic", allow_cloud_llm=False)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-fake-key-for-gate-test"}):
            provider = get_llm_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_anthropic_blocked_when_key_absent(self):
        cfg = _cfg(llm_backend="anthropic", allow_cloud_llm=True)
        env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            provider = get_llm_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_anthropic_provider_returned_when_gate_open(self):
        from providers.anthropic_provider import AnthropicProvider
        cfg = _cfg(llm_backend="anthropic", allow_cloud_llm=True)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-fake-key-open-gate"}):
            with patch("providers.anthropic_provider.AnthropicProvider", AnthropicProvider):
                try:
                    provider = get_llm_provider(cfg)
                    assert isinstance(provider, AnthropicProvider)
                except ImportError:
                    pytest.skip("anthropic SDK not installed; gate open path not testable")

    def test_unknown_backend_defaults_to_ollama(self):
        cfg = _cfg(llm_backend="gpt4all", allow_cloud_llm=True)
        provider = get_llm_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_anthropic_client_never_instantiated_when_gate_closed(self):
        mock_class = MagicMock()
        cfg = _cfg(llm_backend="anthropic", allow_cloud_llm=False)
        with patch("providers.anthropic_provider.AnthropicProvider", mock_class):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-fake"}):
                provider = get_llm_provider(cfg)
        mock_class.assert_not_called()
        assert isinstance(provider, OllamaProvider)


class TestOpenAICompatibleGate:
    def test_blocked_when_gate_closed(self):
        cfg = _cfg(
            llm_backend="openai_compatible",
            allow_cloud_llm=False,
            openai_compatible_base_url="https://api.groq.com/openai/v1",
        )
        mock_class = MagicMock()
        with patch(
            "providers.openai_compatible_provider.OpenAICompatibleProvider", mock_class
        ):
            with patch.dict(os.environ, {"OPENAI_COMPATIBLE_API_KEY": "gsk-fake"}):
                provider = get_llm_provider(cfg)
        mock_class.assert_not_called()
        assert isinstance(provider, OllamaProvider)

    def test_blocked_when_key_absent(self):
        cfg = _cfg(
            llm_backend="openai_compatible",
            allow_cloud_llm=True,
            openai_compatible_base_url="https://api.groq.com/openai/v1",
            openai_compatible_api_key="",
        )
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_COMPATIBLE_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            provider = get_llm_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_blocked_when_base_url_absent(self):
        cfg = _cfg(
            llm_backend="openai_compatible",
            allow_cloud_llm=True,
            openai_compatible_base_url="",
            openai_compatible_api_key="",
        )
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL")
        }
        env["OPENAI_COMPATIBLE_API_KEY"] = "gsk-fake"
        with patch.dict(os.environ, env, clear=True):
            provider = get_llm_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_provider_returned_when_gate_open(self):
        from providers.openai_compatible_provider import OpenAICompatibleProvider
        cfg = _cfg(
            llm_backend="openai_compatible",
            allow_cloud_llm=True,
            openai_compatible_base_url="https://api.groq.com/openai/v1",
        )
        with patch.dict(os.environ, {"OPENAI_COMPATIBLE_API_KEY": "gsk-fake-open"}):
            provider = get_llm_provider(cfg)
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.healthcheck() is True

    def test_strict_raises_when_gate_closed(self):
        cfg = _cfg(llm_backend="openai_compatible", allow_cloud_llm=False)
        with pytest.raises(BackendMisconfigured):
            get_llm_provider(cfg, strict=True)


class TestResolveRoleModels:
    def test_local_defaults(self):
        cfg = _cfg(llm_backend="ollama", allow_cloud_llm=False)
        roles = resolve_role_models(cfg)
        assert roles.generator == cfg.generator_model
        assert roles.prompt == cfg.prompt_model

    def test_anthropic_roles(self):
        cfg = _cfg(llm_backend="anthropic", allow_cloud_llm=True)
        roles = resolve_role_models(cfg)
        assert roles.generator == cfg.anthropic_generator_model
        assert roles.prompt == cfg.anthropic_prompt_model

    def test_openai_compatible_roles(self):
        cfg = _cfg(llm_backend="openai_compatible", allow_cloud_llm=True)
        roles = resolve_role_models(cfg)
        assert roles.generator == cfg.openai_compatible_generator_model
        assert roles.prompt == cfg.openai_compatible_prompt_model
        assert roles.verifier == cfg.openai_compatible_verifier_model
        assert roles.fallback == cfg.openai_compatible_fallback_model

    def test_openai_compatible_default_stack_splits_sizes(self, monkeypatch):
        # GateGuard: callers pytest. Affected API: default Groq role split.
        # Data schemas: RoleModels. User: Groq org limits (8K TPM).
        monkeypatch.setenv("LLM_BACKEND", "openai_compatible")
        monkeypatch.setenv("ALLOW_CLOUD_LLM", "true")
        monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://api.groq.com/openai/v1")
        monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "gsk-test")
        monkeypatch.delenv("OPENAI_COMPATIBLE_GENERATOR_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_COMPATIBLE_FALLBACK_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_COMPATIBLE_VERIFIER_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_COMPATIBLE_PROMPT_MODEL", raising=False)
        from config import Config
        from providers.roles import resolve_role_models

        cfg = Config()
        roles = resolve_role_models(cfg)
        assert roles.generator == "openai/gpt-oss-20b"
        assert roles.fallback == "openai/gpt-oss-120b"
        assert roles.verifier == "openai/gpt-oss-20b"
        assert roles.prompt == "openai/gpt-oss-20b"

    def test_cloud_backend_without_gate_uses_local(self):
        cfg = _cfg(llm_backend="openai_compatible", allow_cloud_llm=False)
        roles = resolve_role_models(cfg)
        assert roles.generator == cfg.generator_model


class TestOllamaProviderDelegation:
    def test_healthcheck_delegates_to_llm_client(self, monkeypatch):
        monkeypatch.setattr(llm_client, "check_ollama_available", lambda: True)
        provider = OllamaProvider()
        assert provider.healthcheck() is True

    def test_generate_delegates_to_ollama_generate(self, monkeypatch):
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
    def test_json_generate_passes_validator_model(self, monkeypatch, valid_analysis_json):
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
    def _make_client(self, tool_input):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = tool_input
        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        return mock_client

    def test_valid_payload_passes_pydantic_validation(self, valid_analysis_json):
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
    def test_generate_returns_content_text_stripped(self):
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


class TestOpenAICompatibleProvider:
    def _make_client(self, tool_args):
        tool_call = MagicMock()
        tool_call.function.arguments = (
            tool_args if isinstance(tool_args, str) else json.dumps(tool_args)
        )
        message = MagicMock()
        message.tool_calls = [tool_call]
        message.content = None
        choice = MagicMock()
        choice.message = message
        mock_response = MagicMock()
        mock_response.choices = [choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_generate_returns_content(self):
        from providers.openai_compatible_provider import OpenAICompatibleProvider
        message = MagicMock()
        message.content = "  hi there  "
        choice = MagicMock()
        choice.message = message
        mock_response = MagicMock()
        mock_response.choices = [choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        provider = OpenAICompatibleProvider(
            api_key="gsk-test", base_url="https://api.groq.com/openai/v1"
        )
        with patch.object(provider, "_client", return_value=mock_client):
            assert provider.generate("openai/gpt-oss-20b", "ping") == "hi there"

    def test_json_generate_validates(self, valid_analysis_json):
        from providers.openai_compatible_provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(
            api_key="gsk-test", base_url="https://api.groq.com/openai/v1"
        )
        with patch.object(
            provider, "_client", return_value=self._make_client(valid_analysis_json)
        ):
            result = provider.json_generate(
                "openai/gpt-oss-20b",
                "sys",
                "user",
                validator_model=AnalysisOutput,
            )
        AnalysisOutput.model_validate(result)

    def test_json_generate_empty_raises(self):
        from providers.openai_compatible_provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(
            api_key="gsk-test", base_url="https://api.groq.com/openai/v1"
        )
        with patch.object(provider, "_client", return_value=self._make_client({})):
            with pytest.raises(ValueError, match="json_schema_validation_failed"):
                provider.json_generate(
                    "openai/gpt-oss-20b",
                    "sys",
                    "user",
                    validator_model=AnalysisOutput,
                )

    def test_json_generate_recovers_failed_generation(self, valid_analysis_json):
        # GateGuard: callers pytest. Affected API: json_generate recovery.
        # Data schemas: AnalysisOutput. User: terminal Groq 401 / draft fail.
        from providers.openai_compatible_provider import OpenAICompatibleProvider

        class FakeAPIError(Exception):
            def __init__(self, body):
                super().__init__("Error code: 400 - tool_use_failed")
                self.body = body

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = FakeAPIError(
            {
                "error": {
                    "message": "tool_use_failed",
                    "failed_generation": json.dumps(valid_analysis_json),
                }
            }
        )
        provider = OpenAICompatibleProvider(
            api_key="gsk-test", base_url="https://api.groq.com/openai/v1"
        )
        with patch.object(provider, "_client", return_value=mock_client):
            result = provider.json_generate(
                "openai/gpt-oss-20b",
                "sys",
                "user",
                validator_model=AnalysisOutput,
            )
        AnalysisOutput.model_validate(result)

    def test_json_generate_accepts_content_json(self, valid_analysis_json):
        from providers.openai_compatible_provider import OpenAICompatibleProvider

        message = MagicMock()
        message.tool_calls = []
        message.content = json.dumps(valid_analysis_json)
        choice = MagicMock()
        choice.message = message
        mock_response = MagicMock()
        mock_response.choices = [choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        provider = OpenAICompatibleProvider(
            api_key="gsk-test", base_url="https://api.groq.com/openai/v1"
        )
        with patch.object(provider, "_client", return_value=mock_client):
            result = provider.json_generate(
                "openai/gpt-oss-20b",
                "sys",
                "user",
                validator_model=AnalysisOutput,
            )
        AnalysisOutput.model_validate(result)


def test_load_config_defaults(monkeypatch):
    # Isolate from developer .env cloud overrides.
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("ALLOW_CLOUD_LLM", "false")
    monkeypatch.delenv("OPENAI_COMPATIBLE_GENERATOR_MODEL", raising=False)
    from config import Config
    cfg = Config()
    assert cfg.llm_backend == "ollama"
    assert cfg.trace_include_text is False
    assert cfg.chunk_overlap < cfg.chunk_size
    assert cfg.openai_compatible_generator_model == "openai/gpt-oss-20b"
    assert cfg.openai_compatible_fallback_model == "openai/gpt-oss-120b"
    assert cfg.openai_compatible_verifier_model == "openai/gpt-oss-20b"
    assert cfg.openai_compatible_prompt_model == "openai/gpt-oss-20b"


def test_config_anthropic_requires_allow_cloud_llm_in_production(monkeypatch):
    from config import Config
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    monkeypatch.setenv("ALLOW_CLOUD_LLM", "false")
    with pytest.raises(ValueError, match="ALLOW_CLOUD_LLM"):
        Config()


def test_config_openai_compatible_requires_allow_cloud_llm_in_production(monkeypatch):
    from config import Config
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("LLM_BACKEND", "openai_compatible")
    monkeypatch.setenv("ALLOW_CLOUD_LLM", "false")
    with pytest.raises(ValueError, match="ALLOW_CLOUD_LLM"):
        Config()


def test_scrub_trace_payload_removes_sensitive_keys():
    from service.tracing import scrub_trace_payload
    payload = {
        "text": "private journal entry",
        "context": "retrieved context",
        "pending_write": {"text": "draft"},
        "latency_ms": 12.5,
        "hits": [{"text": "hit one", "score": 0.9}],
    }
    scrubbed = scrub_trace_payload(payload, include_text=False)
    assert scrubbed["text"] == "[scrubbed]"
    assert scrubbed["context"] == "[scrubbed]"
    assert scrubbed["pending_write"] == "[scrubbed]"
    assert scrubbed["latency_ms"] == 12.5
    assert scrubbed["hits"][0]["text"] == "[scrubbed]"
    assert scrubbed["hits"][0]["score"] == 0.9
