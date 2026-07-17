from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from providers.base import LLMProvider

T = TypeVar("T", bound=BaseModel)


class OllamaProvider(LLMProvider):
    """
    Ollama backend — wraps the existing llm_client functions.

    This is a behavior-preserving adapter: every call delegates to
    llm_client.ollama_generate / llm_client.json_generate so the
    existing retry/parse/validation logic is reused without duplication.
    All existing tests that monkeypatch llm_client.ollama_generate
    continue to work unchanged.
    """

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: int = 30,
    ) -> str:
        from llm_client import ollama_generate

        return ollama_generate(
            model,
            prompt,
            timeout=timeout,
            system=system,
            temperature=temperature,
        )

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
        from llm_client import json_generate as _json_generate

        return _json_generate(
            model,
            system_prompt,
            user_prompt,
            max_retries=max_retries,
            json_schema=json_schema,
            temperature=temperature,
            validator_model=validator_model,
        )

    def healthcheck(self) -> bool:
        from llm_client import check_ollama_available

        return check_ollama_available()
