"""
Centralized Ollama LLM client for multi-model support.
Enhanced with JSON schema enforcement for reliable parsing.
"""

import requests
import json
import logging
import re
from typing import Dict, Optional, Any

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


def json_generate(model: str, system_prompt: str, user_prompt: str, max_retries: int = 5, json_schema: Optional[Dict] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
    """
    Generate and parse JSON response from Ollama with schema enforcement.
    Retries on parse errors with increasingly strict prompts.
    
    Args:
        model: Model name
        system_prompt: System prompt (should instruct JSON format)
        user_prompt: User prompt
        max_retries: Maximum retry attempts (default 5)
        json_schema: Optional JSON schema for format enforcement
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If JSON cannot be parsed after retries
    """
    # Use schema if provided, otherwise use "json" format
    format_spec = json_schema if json_schema else "json"
    
    # Enhance system prompt for strict JSON
    strict_system = system_prompt
    if json_schema:
        strict_system += "\n\nCRITICAL: Return ONLY valid JSON that matches the provided schema. Do not add markdown, no commentary, no code fences. The response must be parseable JSON."
    else:
        strict_system += "\n\nCRITICAL: Return ONLY valid JSON. Do not add markdown, no commentary, no code fences."
    
    for attempt in range(max_retries):
        try:
            # On later attempts, make prompt even stricter
            if attempt >= 2:
                enhanced_user = user_prompt + "\n\nREMINDER: Return ONLY the JSON object. No extra text before or after."
            else:
                enhanced_user = user_prompt
            
            response_text = ollama_generate(model, enhanced_user, system=strict_system, format_spec=format_spec, temperature=temperature)
            
            # Clean response
            response_text = response_text.strip()
            
            # Remove markdown code fences if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            
            # Try direct parse first
            try:
                parsed = json.loads(response_text)
                return parsed
            except json.JSONDecodeError:
                # Try extracting JSON substring
                json_substring = extract_json_substring(response_text)
                parsed = json.loads(json_substring)
                return parsed
        
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                # Log only attempt count and error type, not content
                logging.warning(f"JSON parse error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                continue
            else:
                logging.error(f"Failed to parse JSON after {max_retries} attempts: {type(e).__name__}")
                raise ValueError(f"Could not parse JSON response after {max_retries} attempts: {type(e).__name__}")
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Generation error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                continue
            else:
                logging.error(f"Generation failed after {max_retries} attempts: {type(e).__name__}")
                raise
    
    raise ValueError("Failed to generate valid JSON after retries")


def check_ollama_available() -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
