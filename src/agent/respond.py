"""Grounded respond + verify path for the Journal Agent.

Ports the quality-pipeline instincts from generator_prompts / verifier_prompts
into free-text agent answers (not the AnalysisOutput JSON schema). Draft with a
strong grounded system prompt, optionally verify, revise when groundedness fails.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .providers.base import LLMProvider
from safety import (
    CRISIS_SUPPORT_MESSAGE,
    is_crisis,
    strip_harsh_text,
    strip_ungrounded_quotes,
)

logger = logging.getLogger(__name__)


AGENT_RESPOND_SYSTEM_PROMPT = """You are an emotionally intelligent journaling assistant answering questions about the user's own journal history.

CRITICAL RULES:
1. NEVER invent journal entries, events, people, or feelings that are not in USER_MESSAGE or RETRIEVED_CONTEXT.
2. ONLY use USER_MESSAGE and RETRIEVED_CONTEXT. If RETRIEVED_CONTEXT is empty, say you do not have matching history yet — do not fabricate it.
3. When you reference past writing, stay faithful to the retrieved text. Prefer short quotes or close paraphrases.
4. Be honest about uncertainty. Use hedged language ("may", "might", "could") when unsure.
5. Do not diagnose or give medical advice. If crisis or self-harm is signalled, acknowledge distress and point toward human support rather than reframing with positivity.
6. Tone: warm, specific, non-judgmental. No blame, dismissal, character judgments, or toxic positivity.
7. Offer, never order. Suggestions should be optional invitations.
8. If the user seems to ask for a write/save, remind them writes require explicit confirmation — do not claim you saved anything.
9. Answer in clear prose (not JSON). Keep it concise: a short reflection, then optional next step.

Be accurate first, then empathetic."""


AGENT_VERIFY_SYSTEM_PROMPT = """You are a fact-checking verifier for journaling assistant answers.

Evaluate the DRAFT_ANSWER against:
- USER_MESSAGE: the user's question or note
- RETRIEVED_CONTEXT: past journal excerpts (may be empty)

Check for:
1. GROUNDEDNESS: claims about the user's past must be supported by RETRIEVED_CONTEXT or USER_MESSAGE
2. HALLUCINATIONS: invented entries, events, or relationships
3. SAFETY: crisis/self-harm handling must not use cheerfulness or toxic positivity
4. TONE: no blame, dismissal, character judgment, or diagnosis

Return ONLY JSON matching the schema. No markdown."""


class AgentVerifyVerdict(BaseModel):
    groundedness_score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    rewrite_required: bool = False
    rewrite_instructions: str = ""


AGENT_VERIFY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "groundedness_score": {"type": "number"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "safety_flags": {"type": "array", "items": {"type": "string"}},
        "rewrite_required": {"type": "boolean"},
        "rewrite_instructions": {"type": "string"},
    },
    "required": [
        "groundedness_score",
        "unsupported_claims",
        "safety_flags",
        "rewrite_required",
        "rewrite_instructions",
    ],
}


def build_respond_user_prompt(user_message: str, retrieved_context: str = "") -> str:
    parts = [
        "Answer the user using only the evidence below.",
        "",
        f"USER_MESSAGE:\n{user_message.strip()}",
        "",
    ]
    if retrieved_context and retrieved_context.strip():
        parts.append(f"RETRIEVED_CONTEXT (from past entries):\n{retrieved_context.strip()}")
        parts.append("")
        parts.append(
            "You may reference RETRIEVED_CONTEXT, but do not invent connections "
            "that are not explicitly supported."
        )
    else:
        parts.append("RETRIEVED_CONTEXT: (none)")
        parts.append("")
        parts.append(
            "No past entries were retrieved. Say so clearly if the question "
            "depends on journal history. Do not invent history."
        )
    parts.append("")
    parts.append("Write a grounded prose answer.")
    return "\n".join(parts)


def build_verify_user_prompt(
    draft_answer: str,
    user_message: str,
    retrieved_context: str = "",
) -> str:
    ctx = retrieved_context.strip() if retrieved_context else "(none)"
    return (
        "Verify this draft answer against the allowed evidence.\n\n"
        f"DRAFT_ANSWER:\n{draft_answer}\n\n"
        f"USER_MESSAGE:\n{user_message}\n\n"
        f"RETRIEVED_CONTEXT:\n{ctx}\n\n"
        "Return ONLY JSON matching the schema."
    )


def build_revision_user_prompt(
    draft_answer: str,
    verdict: Dict[str, Any],
    user_message: str,
    retrieved_context: str = "",
) -> str:
    ctx = retrieved_context.strip() if retrieved_context else "(none)"
    instructions = verdict.get("rewrite_instructions") or (
        "Remove unsupported claims and stay strictly within USER_MESSAGE and RETRIEVED_CONTEXT."
    )
    return (
        "Revise the draft answer using the verifier instructions.\n\n"
        f"USER_MESSAGE:\n{user_message}\n\n"
        f"RETRIEVED_CONTEXT:\n{ctx}\n\n"
        f"DRAFT_ANSWER:\n{draft_answer}\n\n"
        f"VERIFIER_SCORE: {verdict.get('groundedness_score')}\n"
        f"UNSUPPORTED_CLAIMS: {verdict.get('unsupported_claims') or []}\n"
        f"SAFETY_FLAGS: {verdict.get('safety_flags') or []}\n"
        f"REWRITE_INSTRUCTIONS: {instructions}\n\n"
        "Return only the revised prose answer. Do not invent new journal history."
    )


def _roles(cfg) -> tuple[str, str, str]:
    """Return (generator, verifier, fallback) model names for the active backend."""
    from providers.roles import resolve_role_models

    roles = resolve_role_models(cfg)
    return roles.generator, roles.verifier, roles.fallback


def generate_grounded_response(
    provider: LLMProvider,
    cfg,
    *,
    user_message: str,
    retrieved_context: str = "",
) -> Dict[str, Any]:
    """Draft → optional verify → revise. Returns answer + quality metadata."""
    # Crisis short-circuit: if user message is crisis, return support message immediately.
    if is_crisis(user_message):
        return {
            "answer": CRISIS_SUPPORT_MESSAGE,
            "verified": False,
            "revised": False,
            "verdict": None,
            "model": "crisis_gate",
        }

    gen_model, ver_model, fallback_model = _roles(cfg)
    user_prompt = build_respond_user_prompt(user_message, retrieved_context)

    try:
        draft = provider.generate(
            gen_model,
            user_prompt,
            system=AGENT_RESPOND_SYSTEM_PROMPT,
            timeout=60,
        ).strip()
    except Exception as e:
        logger.warning("Agent draft failed (%s): %s", gen_model, e)
        if fallback_model != gen_model:
            try:
                draft = provider.generate(
                    fallback_model,
                    user_prompt,
                    system=AGENT_RESPOND_SYSTEM_PROMPT,
                    timeout=60,
                ).strip()
            except Exception as e2:
                logger.warning("Agent fallback draft failed: %s", e2)
                draft = ""
        else:
            draft = ""

    if not draft:
        fallback = (
            retrieved_context.strip()
            if retrieved_context and retrieved_context.strip()
            else "I could not reach the model backend, and no retrieved journal context is available."
        )
        return {
            "answer": fallback,
            "verified": False,
            "revised": False,
            "verdict": None,
            "model": gen_model,
        }

    # Single-pass path when quality mode is off.
    if not getattr(cfg, "quality_mode_default", True):
        answer = strip_harsh_text(draft)
        evidence = user_message + "\n" + (retrieved_context or "")
        answer = strip_ungrounded_quotes(answer, evidence)
        return {
            "answer": answer,
            "verified": False,
            "revised": False,
            "verdict": None,
            "model": gen_model,
        }

    verdict: Optional[Dict[str, Any]] = None
    try:
        verdict = provider.json_generate(
            ver_model,
            AGENT_VERIFY_SYSTEM_PROMPT,
            build_verify_user_prompt(draft, user_message, retrieved_context),
            json_schema=AGENT_VERIFY_SCHEMA,
            max_retries=3,
            validator_model=AgentVerifyVerdict,
        )
    except Exception as e:
        logger.warning("Agent verify failed: %s", e)
        answer = strip_harsh_text(draft)
        evidence = user_message + "\n" + (retrieved_context or "")
        answer = strip_ungrounded_quotes(answer, evidence)
        return {
            "answer": answer,
            "verified": False,
            "revised": False,
            "verdict": None,
            "model": gen_model,
        }

    threshold = float(getattr(cfg, "groundedness_threshold", 0.75) or 0.75)
    score = float(verdict.get("groundedness_score") or 0.0)
    needs_rewrite = bool(verdict.get("rewrite_required")) or score < threshold or bool(
        verdict.get("unsupported_claims")
    ) or bool(verdict.get("safety_flags"))

    if not needs_rewrite:
        answer = draft
        # Apply safety filters
        answer = strip_harsh_text(answer)
        evidence = user_message + "\n" + retrieved_context
        answer = strip_ungrounded_quotes(answer, evidence)
        return {
            "answer": answer,
            "verified": True,
            "revised": False,
            "verdict": verdict,
            "model": gen_model,
        }

    try:
        revised = provider.generate(
            fallback_model,
            build_revision_user_prompt(draft, verdict, user_message, retrieved_context),
            system=AGENT_RESPOND_SYSTEM_PROMPT,
            timeout=60,
        ).strip()
    except Exception as e:
        logger.warning("Agent revise failed: %s", e)
        revised = ""

    answer = revised or draft
    # Apply safety filters
    answer = strip_harsh_text(answer)
    evidence = user_message + "\n" + retrieved_context
    answer = strip_ungrounded_quotes(answer, evidence)

    return {
        "answer": answer,
        "verified": True,
        "revised": bool(revised),
        "verdict": verdict,
        "model": fallback_model if revised else gen_model,
    }
