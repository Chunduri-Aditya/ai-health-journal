"""
Shared pytest fixtures for the AI Health Journal test suite.

Keep fixtures cheap and deterministic; anything that requires Ollama
belongs behind the `integration` marker instead.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pytest

# Make the repo root importable as a module root for tests.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture
def valid_analysis_json() -> Dict[str, Any]:
    """A minimal AnalysisOutput-valid payload used across tests."""
    return {
        "summary": "Felt anxious after the argument with a friend.",
        "emotions": ["anxious", "hurt"],
        "patterns": ["avoidance of confrontation"],
        "triggers": ["argument"],
        "coping_suggestions": [
            "Take a few slow breaths before responding.",
            "Write down the feelings without judgement.",
        ],
        "quotes_from_user": ["I argued with my friend"],
        "confidence": 0.7,
    }


@pytest.fixture
def valid_verifier_json() -> Dict[str, Any]:
    """A minimal VerifierVerdict-valid payload."""
    return {
        "groundedness_score": 0.88,
        "unsupported_claims": [],
        "safety_flags": [],
        "rewrite_required": False,
        "rewrite_instructions": "",
    }
