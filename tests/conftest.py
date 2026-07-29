"""
Shared pytest fixtures for the AI Health Journal test suite.

Keep fixtures cheap and deterministic; anything that requires Ollama
belongs behind the `integration` marker instead.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Dict

import pytest

# Make the repo root importable as a module root for tests.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Redirect Chroma at IMPORT time, not from a fixture.
#
# app.py builds its vector store at module level (`vector_store =
# get_vector_store()`), which executes when a test module imports app -- during
# collection, before any fixture has run. A session-scoped autouse fixture is
# therefore too late: the store is already pointed at the real journal.
# conftest.py is imported before collection, so this is the earliest hook that
# actually covers that path.
_TEST_CHROMA_DIR = tempfile.mkdtemp(prefix="aihj_test_chroma_")
os.environ["CHROMA_PERSIST_DIR"] = _TEST_CHROMA_DIR


@pytest.fixture(scope="session", autouse=True)
def _isolate_chroma_store():
    """Belt-and-braces re-assert of the import-time redirect above.

    Kept because a test that legitimately monkeypatches CHROMA_PERSIST_DIR for
    its own directory could otherwise leave it pointing somewhere unexpected for
    whatever runs next.

    ChromaStore defaults CHROMA_PERSIST_DIR to ./storage/chroma, so any test
    touching real Chroma wrote fixture text straight into the user's journal.
    Measured damage before this existed: 331 collections holding 280 entries of
    which only 23 texts were unique, and adversarial fixtures such as
    "Sometimes I think about harming myself" surfacing as retrieval results for
    unrelated queries in the running app.

    Autouse and session-scoped so isolation does not depend on remembering to
    opt in, and applies however pytest is invoked. The Makefile sets the same
    variable, but that only protects `make test`; a bare `pytest` bypasses it
    entirely, which is exactly how the pollution kept coming back.

    Tests needing their own directory still monkeypatch CHROMA_PERSIST_DIR
    per-test; that overrides this and is unaffected.
    """
    os.environ["CHROMA_PERSIST_DIR"] = _TEST_CHROMA_DIR
    yield


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
