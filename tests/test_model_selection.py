"""Tests for the chat-model candidate filter used by the model dropdown.

Code-generation models are technically valid Ollama chat models but a poor
fit for emotionally-aware journaling. They were previously selectable
alongside real candidates (phi3, qwen3, etc.), including in the manual model
dropdown where a user could pick one directly for analysis.
"""

import pytest

from model_selection import _is_candidate_chat_model


@pytest.mark.parametrize(
    "name",
    [
        "codellama:7b",
        "codellama:latest",
        "deepseek-coder:6.7b",
        "starcoder2:15b",
        "codegemma:7b",
        "nomic-embed-text:latest",
        "bge-large:latest",
    ],
)
def test_non_chat_models_excluded(name):
    assert _is_candidate_chat_model(name) is False


@pytest.mark.parametrize(
    "name",
    [
        "phi3:3.8b",
        "qwen2.5:14b-instruct",
        "samantha-mistral:7b",
        "mistral:7b",
        "llama3.1:8b",
        # A reasoning model, not deepseek-CODER -- must not be caught by the
        # "coder" substring check.
        "deepseek-r1:8b",
    ],
)
def test_chat_models_still_included(name):
    assert _is_candidate_chat_model(name) is True
