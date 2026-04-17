"""
Centralized Ollama LLM client for multi-model support.
Enhanced with JSON schema enforcement for reliable parsing.
"""

import requests
import json
import logging
import re
from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
TIMEOUT_SECONDS = 30

# JSON schemas for structured outputs
DRAFT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "emotions": {"type": "array", "items": {"type": "string"}},
        "patterns": {"type": "array", "items": {"type": "string"}},
        "triggers": {"type": "array", "items": {"type": "string"}},
        "coping_suggestions": {"type": "array", "items": {"type": "string"}},
        "quotes_from_user": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["summary", "emotions", "patterns", "triggers", "coping_suggestions", "quotes_from_user", "confidence"],
    "additionalProperties": False
}

VERIFIER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "groundedness_score": {"type": "number", "minimum": 0, "maximum": 1},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "safety_flags": {"type": "array", "items": {"type": "string"}},
        "rewrite_required": {"type": "boolean"},
        "rewrite_instructions": {"type": "string"}
    },
    "required": ["groundedness_score", "unsupported_claims", "safety_flags", "rewrite_required", "rewrite_instructions"],
    "additionalProperties": False
}


def ollama_generate(model: str, prompt: str, timeout: int = TIMEOUT_SECONDS, system: Optional[str] = None, format_spec: Optional[Dict] = None, temperature: Optional[float] = None) -> str:
    """
    Generate text from Ollama API.
    
    Args:
        model: Model name (e.g., "phi3:3.8b")
        prompt: User prompt
        timeout: Request timeout in seconds
        system: Optional system prompt
        format_spec: Optional format specification (JSON schema or "json")
        
    Returns:
        Generated text response
        
    Raises:
        requests.exceptions.RequestException: On API errors
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature if temperature is not None else 0.0,  # Allow higher temp for baseline
            "top_p": 1.0
        }
    }
    if system:
        payload["system"] = system
    if format_spec:
        payload["format"] = format_spec
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=timeout
        )
        
        if response.status_code != 200:
            logging.warning(f"Ollama API error {response.status_code}")
            raise requests.exceptions.RequestException(f"Ollama API returned {response.status_code}")
        
        result = response.json()
        return result.get("response", "").strip()
    
    except requests.exceptions.Timeout:
        logging.error(f"Ollama API timeout for model {model}")
        raise
    except Exception as e:
        logging.error(f"Ollama API error: {e}")
        raise


def extract_json_substring(text: str) -> str:
    """
    Extract JSON substring from text that may contain extra content.
    Finds first '{' and last '}' and extracts everything between.
    """
    first_brace = text.find('{')
    if first_brace == -1:
        raise ValueError("No opening brace found")
    
    last_brace = text.rfind('}')
    if last_brace == -1 or last_brace <= first_brace:
        raise ValueError("No closing brace found")
    
    return text[first_brace:last_brace + 1]


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` fences if present."""
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    return text


def _parse_json_lenient(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON from model output, tolerating markdown fences and
    pre/postamble around a single JSON object.

    Raises json.JSONDecodeError (or ValueError from extract_json_substring)
    when both the direct parse and substring extraction fail.
    """
    cleaned = _strip_markdown_fences(response_text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_substring = extract_json_substring(cleaned)
        return json.loads(json_substring)


def json_generate(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 5,
    json_schema: Optional[Dict] = None,
    temperature: Optional[float] = None,
    *,
    validator_model: Optional[Type[T]] = None,
    return_model: bool = False,
) -> Union[Dict[str, Any], T]:
    """
    Generate and parse JSON response from Ollama with schema enforcement.
    Retries on parse errors (and, when validator_model is supplied, on
    schema-validation errors) with increasingly strict prompts.

    Args:
        model: Model name.
        system_prompt: System prompt (should instruct JSON format).
        user_prompt: User prompt.
        max_retries: Maximum retry attempts (default 5).
        json_schema: Optional JSON schema for Ollama's ``format`` parameter.
        temperature: Sampling temperature (None -> 0.0).
        validator_model: Optional pydantic model. When supplied, the parsed
            JSON is validated against it and retries also re-attempt on
            ``pydantic.ValidationError``.
        return_model: When True and validator_model is supplied, return the
            validated pydantic instance instead of a plain dict.

    Returns:
        Parsed JSON as ``dict`` by default. If ``validator_model`` is set and
        ``return_model`` is True, returns an instance of ``validator_model``.
        If ``validator_model`` is set and ``return_model`` is False, returns
        the validated-then-dumped dict (guaranteed to satisfy the model).

    Raises:
        ValueError: On parse failure after all retries
            (``"json_parse_failed: <type>"``) or on validation failure after
            all retries (``"json_schema_validation_failed: <summary>"``).
    """
    format_spec = json_schema if json_schema else "json"

    strict_system = system_prompt
    if json_schema:
        strict_system += "\n\nCRITICAL: Return ONLY valid JSON that matches the provided schema. Do not add markdown, no commentary, no code fences. The response must be parseable JSON."
    else:
        strict_system += "\n\nCRITICAL: Return ONLY valid JSON. Do not add markdown, no commentary, no code fences."

    last_validation_error: Optional[ValidationError] = None
    last_parse_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            if attempt >= 2:
                enhanced_user = (
                    user_prompt
                    + "\n\nREMINDER: Return ONLY the JSON object. No extra text before or after."
                )
            else:
                enhanced_user = user_prompt

            response_text = ollama_generate(
                model,
                enhanced_user,
                system=strict_system,
                format_spec=format_spec,
                temperature=temperature,
            )

            try:
                parsed = _parse_json_lenient(response_text)
            except (json.JSONDecodeError, ValueError) as parse_err:
                last_parse_error = parse_err
                if attempt < max_retries - 1:
                    logging.warning(
                        "JSON parse error (attempt %d/%d): %s",
                        attempt + 1, max_retries, type(parse_err).__name__,
                    )
                    continue
                logging.error(
                    "Failed to parse JSON after %d attempts: %s",
                    max_retries, type(parse_err).__name__,
                )
                raise ValueError(
                    f"json_parse_failed: {type(parse_err).__name__}"
                ) from parse_err

            if validator_model is None:
                return parsed

            try:
                validated = validator_model.model_validate(parsed)
            except ValidationError as ve:
                last_validation_error = ve
                if attempt < max_retries - 1:
                    # Log field paths only; keep values out of logs.
                    bad_fields = ",".join(
                        ".".join(str(p) for p in err.get("loc", ())) for err in ve.errors()
                    )
                    logging.warning(
                        "Schema validation error (attempt %d/%d): %s",
                        attempt + 1, max_retries, bad_fields or "unknown",
                    )
                    continue
                logging.error(
                    "Schema validation failed after %d attempts", max_retries
                )
                summary = ",".join(
                    ".".join(str(p) for p in err.get("loc", ())) for err in ve.errors()
                )
                raise ValueError(
                    f"json_schema_validation_failed: {summary or 'unknown'}"
                ) from ve

            return validated if return_model else validated.model_dump()

        except ValueError:
            # Already formatted above; bubble up.
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(
                    "Generation error (attempt %d/%d): %s",
                    attempt + 1, max_retries, type(e).__name__,
                )
                continue
            logging.error(
                "Generation failed after %d attempts: %s",
                max_retries, type(e).__name__,
            )
            raise

    # Unreachable in practice; all paths above raise or return.
    if last_validation_error is not None:
        raise ValueError("json_schema_validation_failed: retries_exhausted")
    if last_parse_error is not None:
        raise ValueError(f"json_parse_failed: {type(last_parse_error).__name__}")
    raise ValueError("Failed to generate valid JSON after retries")


def check_ollama_available() -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
