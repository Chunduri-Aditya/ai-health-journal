# GateGuard fact: callers agent/respond.py, agent/tools.py, service/main.py, app.py. User: Implement the plan as specified. Do NOT edit the plan file.
"""Safety package: crisis detection, tone policing, grounding checks, redaction."""

from __future__ import annotations

from .crisis import (
    CRISIS_SUPPORT_MESSAGE,
    DISTRESS_STEADYING_MESSAGE,
    is_crisis,
    is_distress,
)
from .grounding import strip_ungrounded_quotes
from .tone import strip_harsh_text

# Re-export redact from privacy
from ..privacy.redact import redact

__all__ = [
    "CRISIS_SUPPORT_MESSAGE",
    "DISTRESS_STEADYING_MESSAGE",
    "is_crisis",
    "is_distress",
    "strip_harsh_text",
    "strip_ungrounded_quotes",
    "redact",
]
