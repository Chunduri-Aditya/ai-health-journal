# GateGuard: callers app.py analyze_entry. Affected API: classify_journal_relevance().
# Data schemas: RelevanceVerdict. User: "simple model to check if journal message is
# actually relevant... if they ask coding questions its kinda stealing".
"""Cheap journal-relevance gate.

Blocks obvious non-journal traffic (coding / homework / general Q&A) before the
Draft → Verify → Revise stack spends Groq TPM. Heuristics run first (zero
tokens); ambiguous short question-like messages can ask the prompt-role model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .providers.base import LLMProvider
from .providers.roles import resolve_role_models

logger = logging.getLogger(__name__)

OFF_TOPIC_MESSAGE = (
    "This space is for journaling how you feel — not coding help or general Q&A. "
    "That keeps the reflection models free for your real entries. "
    "Write a bit about your day or how you're feeling, and I'll meet you there."
)

_CODE_FENCE = re.compile(r"```")
_CODING_PHRASE = re.compile(
    r"(?i)\b("
    r"write\s+(a\s+)?(python|javascript|java|rust|go|sql|html|css)\b|"
    r"write\s+(me\s+)?(a\s+)?(function|class|script|regex|query)\b|"
    r"debug\s+(this|my)\b|"
    r"fix\s+(this|my)\s+code\b|"
    r"leetcode|hackerrank|"
    r"stack\s*overflow|"
    r"implement\s+(a\s+)?(function|class|api|endpoint)\b|"
    r"console\.log|System\.out\.println|"
    r"npm\s+install|pip\s+install|"
    r"\bdef\s+\w+\s*\(|\bfunction\s+\w+\s*\(|\bclass\s+\w+\s*[{:]|"
    r"SELECT\s+.+\s+FROM\b|"
    r"how\s+do\s+i\s+(code|program|implement|debug|compile)\b"
    r")"
)
_IMPORTISH = re.compile(
    r"(?m)^(?:import\s+\w+|from\s+\w+\s+import\s+|using\s+\w+|package\s+\w+)"
)

_JOURNALISH = re.compile(
    r"(?i)\b("
    r"i\s+(feel|felt|am\s+feeling|noticed|realized|grateful|worried|anxious|"
    r"tired|proud|sad|happy|angry|overwhelmed|stressed)|"
    r"today\s+i|this\s+morning|last\s+night|my\s+day|"
    r"journal|grateful|gratitude|reflection|emotions?"
    r")\b"
)

_QUESTIONISH = re.compile(r"\?")

RELEVANCE_SYSTEM = """You gate a private health/journaling app.
Decide if the user text is a personal journal entry or emotional reflection
suitable for supportive journaling analysis.

Mark relevant=false for: coding help, homework, general knowledge Q&A,
math/debugging, generating software, or anything that is not about the
user's lived experience / feelings / day.

Mark relevant=true for: feelings, day events, stress, gratitude, relationships,
health symptoms in a personal narrative, or short emotional check-ins.

Return ONLY JSON matching the schema."""

RELEVANCE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "category": {
            "type": "string",
            "enum": ["journal", "coding", "qa", "other"],
        },
    },
    "required": ["relevant", "category"],
}


@dataclass(frozen=True)
class RelevanceVerdict:
    relevant: bool
    category: str
    source: str  # heuristic | llm | fail_open


def _heuristic(text: str) -> Optional[RelevanceVerdict]:
    raw = (text or "").strip()
    if not raw:
        return RelevanceVerdict(False, "other", "heuristic")

    if _CODE_FENCE.search(raw) or _CODING_PHRASE.search(raw) or _IMPORTISH.search(raw):
        return RelevanceVerdict(False, "coding", "heuristic")

    if _JOURNALISH.search(raw) and not _QUESTIONISH.search(raw):
        return RelevanceVerdict(True, "journal", "heuristic")

    if len(raw) < 180 and _QUESTIONISH.search(raw) and not _JOURNALISH.search(raw):
        if re.search(
            r"(?i)\b(how|what|why|can you|could you|explain|write|generate)\b", raw
        ):
            return RelevanceVerdict(False, "qa", "heuristic")

    return None


def classify_journal_relevance(
    text: str,
    *,
    provider: Optional[LLMProvider] = None,
    cfg: Any = None,
    use_llm: bool = True,
) -> RelevanceVerdict:
    """Return whether ``text`` should enter the journaling analysis pipeline."""
    hit = _heuristic(text)
    if hit is not None:
        return hit

    if not use_llm or provider is None or cfg is None:
        return RelevanceVerdict(True, "journal", "fail_open")

    roles = resolve_role_models(cfg)
    try:
        result = provider.json_generate(
            roles.prompt,
            RELEVANCE_SYSTEM,
            f"Classify this user text:\n\n{text.strip()}",
            json_schema=RELEVANCE_SCHEMA,
            max_retries=2,
        )
        relevant = bool(result.get("relevant"))
        category = result.get("category") or ("journal" if relevant else "other")
        if category not in ("journal", "coding", "qa", "other"):
            category = "other"
        return RelevanceVerdict(relevant, category, "llm")
    except Exception as e:
        logger.warning("Relevance LLM gate failed open: %s", e)
        return RelevanceVerdict(True, "journal", "fail_open")
