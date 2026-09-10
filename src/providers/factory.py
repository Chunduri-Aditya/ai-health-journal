# GateGuard: callers app.py, service/main.py, agent/graph.py, tests/test_providers.py.
# Affected API: get_llm_provider(cfg, strict=) now supports openai_compatible.
# Data schemas: none.
# User: Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.
from __future__ import annotations

import logging
import os

from .base import LLMProvider

logger = logging.getLogger(__name__)

_CLOUD_BACKENDS = frozenset({"anthropic", "openai_compatible"})


class BackendMisconfigured(RuntimeError):
    """Raised when a configured cloud backend cannot be constructed."""


def get_llm_provider(cfg, *, strict: bool = False) -> LLMProvider:
    """
    Return the appropriate LLMProvider based on configuration.

    Gate logic — mirrors vector_store/factory.py::get_vector_store:
    - Default (LLM_BACKEND=ollama): always returns OllamaProvider.
    - Cloud paths (anthropic | openai_compatible): require ALLOW_CLOUD_LLM=true.
      If the gate is closed and strict=False, returns OllamaProvider and logs.
      If strict=True, raises BackendMisconfigured instead of falling back.

    Privacy posture: no network call leaves localhost unless
    LLM_BACKEND is a cloud backend AND ALLOW_CLOUD_LLM=true.
    """
    backend = (cfg.llm_backend or "ollama").lower()

    if backend not in _CLOUD_BACKENDS:
        from .ollama_provider import OllamaProvider
        return OllamaProvider()

    def _fallback(reason: str) -> LLMProvider:
        if strict:
            raise BackendMisconfigured(
                f"LLM_BACKEND={backend} but provider could not be constructed: {reason}"
            )
        logger.warning("%s Falling back to Ollama.", reason)
        from .ollama_provider import OllamaProvider
        return OllamaProvider()

    if not cfg.allow_cloud_llm:
        return _fallback(
            f"LLM_BACKEND={backend} but ALLOW_CLOUD_LLM=false. "
            "Cloud LLM is not permitted. Set ALLOW_CLOUD_LLM=true to enable."
        )

    if backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return _fallback(
                "LLM_BACKEND=anthropic and ALLOW_CLOUD_LLM=true but "
                "ANTHROPIC_API_KEY is not set."
            )
        try:
            from .anthropic_provider import AnthropicProvider
            return AnthropicProvider(api_key=api_key)
        except ImportError:
            return _fallback(
                "The 'anthropic' package is not installed. "
                "Install with: pip install anthropic."
            )

    # openai_compatible — dedicated key only (do not reuse OPENAI_API_KEY for chat).
    api_key = (
        os.environ.get("OPENAI_COMPATIBLE_API_KEY", "").strip()
        or getattr(cfg, "openai_compatible_api_key", "")
        or ""
    )
    base_url = (
        getattr(cfg, "openai_compatible_base_url", "")
        or os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "")
    ).strip()
    if not api_key:
        return _fallback(
            "LLM_BACKEND=openai_compatible and ALLOW_CLOUD_LLM=true but "
            "OPENAI_COMPATIBLE_API_KEY is not set."
        )
    if not base_url:
        return _fallback(
            "LLM_BACKEND=openai_compatible and ALLOW_CLOUD_LLM=true but "
            "OPENAI_COMPATIBLE_BASE_URL is not set "
            "(e.g. https://api.groq.com/openai/v1)."
        )
    try:
        from .openai_compatible_provider import OpenAICompatibleProvider
        return OpenAICompatibleProvider(api_key=api_key, base_url=base_url)
    except ImportError:
        return _fallback(
            "The 'openai' package is not installed. Install with: pip install openai."
        )
    except ValueError as e:
        return _fallback(str(e))
