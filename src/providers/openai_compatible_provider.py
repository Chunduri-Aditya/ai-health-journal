# GateGuard: callers providers/factory.py, tests/test_providers.py.
# Affected API: LLMProvider generate/json_generate/healthcheck via OpenAI SDK + base_url.
# Data schemas: none new (reuses caller json_schema / validator_model).
# User: Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.
"""OpenAI-compatible chat backend (Groq, OpenRouter, etc.).

Uses the ``openai`` SDK with a custom ``base_url``. Forced tool calling
mirrors AnthropicProvider so json_generate returns structured dicts without
the Ollama markdown/retry ladder.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .base import LLMProvider

T = TypeVar("T", bound=BaseModel)

_JSON_TOOL_NAME = "structured_output"


def _parse_json_object(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty JSON payload")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON payload must be an object")
    return parsed


def _recover_failed_generation(exc: BaseException) -> Optional[Dict[str, Any]]:
    """Groq tool_use_failed often embeds the intended JSON in failed_generation."""
    body: Any = None
    for attr in ("body", "response"):
        candidate = getattr(exc, attr, None)
        if candidate is None:
            continue
        if hasattr(candidate, "json") and callable(candidate.json):
            try:
                body = candidate.json()
            except Exception:
                body = None
        elif isinstance(candidate, dict):
            body = candidate
        if body:
            break
    if not isinstance(body, dict):
        # openai.BadRequestError string often contains the JSON blob.
        text = str(exc)
        marker = "failed_generation"
        if marker not in text:
            return None
        try:
            # Prefer parsing from error dict embedded in the message if present.
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            outer = json.loads(text[start : end + 1])
            body = outer
        except Exception:
            return None

    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        return None
    failed = err.get("failed_generation")
    if not isinstance(failed, str) or not failed.strip():
        return None
    try:
        return _parse_json_object(failed)
    except Exception:
        return None


class OpenAICompatibleProvider(LLMProvider):
    """Chat completions via any OpenAI-compatible HTTP API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_COMPATIBLE_API_KEY is not set. "
                "Set the environment variable or pass api_key explicitly."
            )
        if not (base_url or "").strip():
            raise ValueError(
                "OPENAI_COMPATIBLE_BASE_URL is not set. "
                "Example for Groq: https://api.groq.com/openai/v1"
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client_instance: Any = None

    def _client(self):
        if self._client_instance is not None:
            return self._client_instance
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for OpenAICompatibleProvider. "
                "Install it with: pip install openai"
            ) from e
        self._client_instance = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client_instance

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
        messages: list[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 2048,
            "timeout": timeout,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return content.strip()

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
        """Structured JSON via forced function/tool calling."""
        client = self._client()
        parameters: Dict[str, Any] = json_schema or {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": _JSON_TOOL_NAME,
                    "description": (
                        "Return the structured analysis as a JSON object "
                        "matching the schema."
                    ),
                    "parameters": parameters,
                },
            }
        ]

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": _JSON_TOOL_NAME},
            },
            "max_tokens": 2048,
            "timeout": 60,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        last_error: Optional[Exception] = None
        for _attempt in range(max(1, max_retries)):
            try:
                response = client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                tool_calls = getattr(message, "tool_calls", None) or []
                if not tool_calls:
                    content = (getattr(message, "content", None) or "").strip()
                    if content:
                        parsed = _parse_json_object(content)
                    else:
                        raise ValueError(
                            "Expected a tool call from the OpenAI-compatible backend; "
                            "none was returned."
                        )
                else:
                    raw_args = tool_calls[0].function.arguments
                    if isinstance(raw_args, dict):
                        parsed = raw_args
                    else:
                        parsed = json.loads(raw_args or "{}")
                if not isinstance(parsed, dict):
                    raise ValueError("Tool call arguments must be a JSON object.")

                if validator_model is None:
                    return parsed

                try:
                    validated = validator_model.model_validate(parsed)
                except ValidationError as ve:
                    summary = ",".join(
                        ".".join(str(p) for p in err.get("loc", ()))
                        for err in ve.errors()
                    )
                    logging.error(
                        "OpenAICompatibleProvider: schema validation failed: %s",
                        summary or "unknown",
                    )
                    raise ValueError(
                        f"json_schema_validation_failed: {summary or 'unknown'}"
                    ) from ve
                return validated.model_dump()
            except ValueError:
                raise
            except Exception as e:
                recovered = _recover_failed_generation(e)
                if recovered is not None:
                    if validator_model is None:
                        return recovered
                    try:
                        return validator_model.model_validate(recovered).model_dump()
                    except ValidationError:
                        pass
                last_error = e
                logging.warning(
                    "OpenAICompatibleProvider.json_generate attempt failed: %s", e
                )

        raise RuntimeError(
            f"OpenAICompatibleProvider.json_generate failed after retries: {last_error}"
        )

    def healthcheck(self) -> bool:
        """Key + base_url present; no live call (avoids burning free-tier quota)."""
        return bool(self._api_key) and bool(self._base_url)
