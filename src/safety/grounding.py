# GateGuard fact: callers agent/respond.py. User: Implement the plan as specified. Do NOT edit the plan file.
"""Grounding checks for free-text agent answers.

Removes quoted phrases not found in the allowed evidence (user message +
retrieved context). Lenient: only catches obvious fabrications, not subtle
paraphrasing.
"""

from __future__ import annotations

import re


def strip_ungrounded_quotes(answer: str, evidence: str) -> str:
    """Remove quoted spans not found in the evidence.

    Keeps lenient: only exact substring matches. A paraphrased reference
    ("I was stressed" vs "felt overwhelmed") is treated as grounded.
    """
    if not answer or not answer.strip():
        return answer
    # Find quoted spans (double or single quotes).
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', answer)
    for double, single in quoted:
        quote = double or single
        if quote and quote not in evidence:
            # Replace the full quote with [REDACTED]
            answer = answer.replace(f'"{quote}"', "[REDACTED]")
            answer = answer.replace(f"'{quote}'", "[REDACTED]")
    return answer
