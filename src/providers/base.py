from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider:
    """
    Abstract base class for LLM provider backends.

    Concrete implementations: OllamaProvider (local, default) and
    AnthropicProvider (cloud, opt-in, gated by ALLOW_CLOUD_LLM).

    Both surfaces share this interface so app.py and the eval harness
    can call either backend without route-level changes.
    """

    # --- primary API ---------------------------------------------------------

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: int = 30,
    ) -> str:
        """
        Free-text generation.

        Returns the model's response as a plain string.
        Raises on API error or timeout.
        """
        raise NotImplementedError

    def json_generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        temperature: Optional[float] = None,
        validator_model: Optional[Type[T]] = None,
    ) -> Dict[str, Any]:
        """
        Structured JSON generation with optional pydantic validation.

        The Ollama path retries with schema-enforced prompts and leniently
        parses markdown fences.  The Anthropic path uses forced tool use
        (tool_choice={"type": "tool"}) to guarantee structured input directly,
        skipping the retry ladder entirely.

        When validator_model is supplied the returned dict is guaranteed to
        satisfy model_validate().  Empty dicts ({}) or missing required fields
        raise ValueError("json_schema_validation_failed: ...") after retries.
        """
        raise NotImplementedError

    def healthcheck(self) -> bool:
        """
        Return True when the backend is reachable and ready to serve requests.

        For Ollama: GET http://localhost:11434 must return 200.
        For Anthropic: ANTHROPIC_API_KEY is present (no token burn on every request).
        """
        raise NotImplementedError
