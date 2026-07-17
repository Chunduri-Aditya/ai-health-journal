from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from providers.base import LLMProvider

T = TypeVar("T", bound=BaseModel)

# Tool name used for forced JSON generation.
_JSON_TOOL_NAME = "structured_output"


class AnthropicProvider(LLMProvider):
    """
    Anthropic (Claude) backend — uses forced tool use for guaranteed JSON.

    Design notes:
    - The ANTHROPIC_API_KEY is read from the environment at construction time.
      It is NEVER logged, stored in session payloads, or written to any cache.
    - For json_generate, tool_choice={"type": "tool", "name": _JSON_TOOL_NAME}
      forces the model to return structured input matching the supplied schema.
      This bypasses the markdown-fence / extract_json_substring / retry ladder
      entirely — the SDK returns the parsed dict directly from message.content[0].input.
    - pydantic validation (validator_model) still runs on both paths so the
      same schema guarantee holds regardless of backend.
    - Free-text generate() goes through the messages API without tool use.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        # Read key from env; never log or expose it.
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Set the environment variable or pass api_key explicitly."
            )

    def _client(self):
        """Lazy import of the anthropic SDK to avoid hard dependency at module load."""
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            ) from e
        return anthropic.Anthropic(api_key=self._api_key)

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: int = 30,
    ) -> str:
        client = self._client()
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = client.messages.create(**kwargs)
        return response.content[0].text.strip()

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
        Structured JSON via forced tool use.

        Anthropic's tool_choice={"type": "tool"} guarantees the model
        always calls the named tool.  The response arrives as a parsed dict
        in message.content[0].input — no markdown stripping, no retry needed
        for parse failures (the SDK raises on malformed tool calls).

        Pydantic validation (validator_model) runs after the call, same as the
        Ollama path (defect #12 fix applies on both backends).
        """
        client = self._client()

        # Build a tool whose input_schema matches the supplied json_schema.
        # If no schema is provided, use a minimal open-ended object schema.
        input_schema: Dict[str, Any] = json_schema or {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }

        tool = {
            "name": _JSON_TOOL_NAME,
            "description": (
                "Return the structured analysis as a JSON object matching the schema."
            ),
            "input_schema": input_schema,
        }

        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": _JSON_TOOL_NAME},
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = client.messages.create(**kwargs)

        # Forced tool use guarantees content[0] is a tool_use block.
        tool_block = response.content[0]
        if tool_block.type != "tool_use":
            raise ValueError(
                f"Expected tool_use block, got {tool_block.type}. "
                "Anthropic may have changed the forced-tool-use behavior."
            )

        parsed: Dict[str, Any] = tool_block.input

        if validator_model is None:
            return parsed

        # Pydantic validation — defect #12 fix on the Anthropic path.
        try:
            validated = validator_model.model_validate(parsed)
        except ValidationError as ve:
            summary = ",".join(
                ".".join(str(p) for p in err.get("loc", ())) for err in ve.errors()
            )
            logging.error(
                "AnthropicProvider: schema validation failed: %s",
                summary or "unknown",
            )
            raise ValueError(
                f"json_schema_validation_failed: {summary or 'unknown'}"
            ) from ve

        return validated.model_dump()

    def healthcheck(self) -> bool:
        """
        Return True when a valid API key is present.

        We do NOT make a live API call here to avoid burning tokens on every
        request.  Key presence is the minimum gate; actual auth errors surface
        on the first real call and are caught by the route exception handler.
        """
        return bool(self._api_key)
