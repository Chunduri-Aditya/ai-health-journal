"""Confirmation gate for agent write actions.

A write tool never mutates the store on first call. It parks a pending
payload on agent state; a second user turn must confirm before apply.
This is the attack surface Agent Shield evals target.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import re
import time

_CONFIRM = re.compile(r"^\s*(yes|y|confirm|approve)\s*[.!]?\s*$", re.I)
_DENY = re.compile(r"^\s*(no|n|cancel|deny|abort|reject)\s*[.!]?\s*$", re.I)


def is_confirmation(text: str) -> bool:
    return bool(_CONFIRM.match(text or ""))


def is_denial(text: str) -> bool:
    return bool(_DENY.match(text or ""))


def build_pending_write(
    *,
    entry_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    namespace: str = "ai-health-journal",
) -> Dict[str, Any]:
    return {
        "action": "add_entry",
        "entry_id": entry_id,
        "text": text,
        "metadata": dict(metadata or {}),
        "namespace": namespace,
        "proposed_at": time.time(),
    }
