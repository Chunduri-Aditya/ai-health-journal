# GateGuard fact: callers agent/graph.py. User: Implement the plan as specified. Do NOT edit the plan file.
"""LLM intent classification for agent routing.

Uses provider.json_generate with a small schema. Falls back to regex route_intent
on failure. The regex is kept as fallback (not removed) so test_agent_routing.py
still works unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from ..providers.base import LLMProvider
from ..providers.roles import resolve_role_models

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a journaling agent.

Given a user message, classify the intent into one of these categories:
- retrieve: user wants to find/search/recall past journal entries
- metadata: user wants to list entries, see entry IDs, or query metadata
- write: user wants to save/store/add/log a new journal entry
- respond: user is asking a question or having a conversation

Return ONLY JSON matching the schema."""

INTENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["retrieve", "metadata", "write", "respond"],
        },
    },
    "required": ["intent"],
}


def classify_intent(
    provider: LLMProvider,
    cfg,
    text: str,
) -> Literal["retrieve", "metadata", "write", "respond"]:
    """LLM-based intent classification.

    Uses the prompt role model from resolve_role_models for fast classification.
    Returns one of: retrieve, metadata, write, respond.
    Raises on failure (caller should catch and use regex fallback).
    """
    prompt = f"Classify the intent of this user message:\n\n{text}"
    roles = resolve_role_models(cfg)
    try:
        result = provider.json_generate(
            roles.prompt,
            INTENT_SYSTEM_PROMPT,
            prompt,
            json_schema=INTENT_SCHEMA,
            max_retries=2,
        )
        intent = result.get("intent")
        if intent in ("retrieve", "metadata", "write", "respond"):
            return intent  # type: ignore[return-value]
        logger.warning("Unexpected intent value: %s", intent)
        raise ValueError(f"Invalid intent: {intent}")
    except Exception as e:
        logger.warning("Intent classification failed: %s", e)
        raise
