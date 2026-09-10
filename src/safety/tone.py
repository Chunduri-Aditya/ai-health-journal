# GateGuard fact: callers agent/respond.py. User: Implement the plan as specified. Do NOT edit the plan file.
"""Harsh tone detection for assistant-authored responses.

Blame, dismissal, and clinical labelling are the three failure modes that turn
a journaling reply into something that lands badly on someone already struggling.
"""

from __future__ import annotations

import re

_HARSH_OUTPUT_PATTERNS = re.compile(
    r"("
    # Blame and command-shame
    r"\byou\s+(?:should|need\s+to|have\s+to|must)\s+(?:just\s+)?(?:stop|quit|get\s+over)\b|"
    r"\bjust\s+(?:get\s+over|move\s+on|snap\s+out)\b|"
    r"\byou(?:'re| are)\s+(?:being\s+)?(?:irrational|dramatic|lazy|weak|childish|ridiculous|overreacting)\b|"
    r"\byour\s+own\s+fault\b|\byou\s+brought\s+this\s+on\s+yourself\b|"
    r"\bstop\s+(?:being|feeling|complaining|whining)\b|"
    # Dismissal / minimisation
    r"\bit'?s\s+not\s+(?:that\s+)?bad\b|"
    r"\bothers\s+have\s+it\s+worse\b|\bcould\s+be\s+worse\b|"
    r"\bstop\s+overthinking\b|"
    # Clinical labelling of the person (diagnosis is out of scope for this app)
    r"\byou\s+(?:have|suffer\s+from)\s+(?:depression|anxiety|a\s+disorder|bipolar|ptsd)\b|"
    r"\byou(?:'re| are)\s+(?:clinically\s+)?(?:depressed|mentally\s+ill)\b"
    r")",
    re.IGNORECASE,
)


def strip_harsh_text(text: str) -> str:
    """Remove sentences containing harsh, blaming, or dismissive phrasing.

    A simple last-resort filter for free-text agent answers. Removes full
    sentences matching the harsh pattern. Preserves empty result when all
    sentences are harsh — caller should decide final behavior.
    """
    if not text or not text.strip():
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not _HARSH_OUTPUT_PATTERNS.search(s)]
    return " ".join(kept)
