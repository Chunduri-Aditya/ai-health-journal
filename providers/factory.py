from __future__ import annotations

import logging

from providers.base import LLMProvider

logger = logging.getLogger(__name__)


def get_llm_provider(cfg) -> LLMProvider:
    """
    Return the appropriate LLMProvider based on configuration.

    Gate logic — mirrors vector_store/factory.py::get_vector_store:
    - Default (LLM_BACKEND=ollama): always returns OllamaProvider.
    - Cloud path (LLM_BACKEND=anthropic): requires ALLOW_CLOUD_LLM=true.
      If the gate is closed, returns OllamaProvider and logs the reason.
      The anthropic SDK client is never instantiated when the gate is closed,
      so ANTHROPIC_API_KEY is never read into memory in that case.

    Privacy posture: no network call leaves localhost unless
    LLM_BACKEND=anthropic AND ALLOW_CLOUD_LLM=true.
    """
    backend = (cfg.llm_backend or "ollama").lower()

    if backend != "anthropic":
        from providers.ollama_provider import OllamaProvider
        return OllamaProvider()

    # Anthropic selected — check the hard gate.
    if not cfg.allow_cloud_llm:
        logger.warning(
            "LLM_BACKEND=anthropic but ALLOW_CLOUD_LLM=false. "
            "Cloud LLM is not permitted. Falling back to Ollama. "
            "Set ALLOW_CLOUD_LLM=true to enable the Anthropic backend."
        )
        from providers.ollama_provider import OllamaProvider
        return OllamaProvider()

    # Gate is open — now read the key and construct the provider.
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning(
            "LLM_BACKEND=anthropic and ALLOW_CLOUD_LLM=true but "
            "ANTHROPIC_API_KEY is not set. Falling back to Ollama."
        )
        from providers.ollama_provider import OllamaProvider
        return OllamaProvider()

    try:
        from providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key)
    except ImportError:
        logger.warning(
            "The 'anthropic' package is not installed. "
            "Install with: pip install anthropic. Falling back to Ollama."
        )
        from providers.ollama_provider import OllamaProvider
        return OllamaProvider()
