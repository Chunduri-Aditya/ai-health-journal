"""Tests for safety integration in agent responses."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.respond import generate_grounded_response
from src.safety import CRISIS_SUPPORT_MESSAGE


class TestSafetyAgentIntegration:
    """Test that safety filters are wired into generate_grounded_response."""

    def test_crisis_short_circuit(self):
        """Crisis message should return support message immediately without calling provider."""
        mock_provider = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.quality_mode_default = True

        crisis_message = "I want to kill myself"
        result = generate_grounded_response(
            mock_provider,
            mock_cfg,
            user_message=crisis_message,
            retrieved_context="",
        )

        # Should return crisis support message
        assert result["answer"] == CRISIS_SUPPORT_MESSAGE
        assert result["model"] == "crisis_gate"
        # Provider should NOT have been called
        mock_provider.generate.assert_not_called()
        mock_provider.json_generate.assert_not_called()

    def test_non_crisis_proceeds_normally(self):
        """Non-crisis message should proceed through normal draft path."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "This is a normal response."
        mock_cfg = MagicMock()
        mock_cfg.quality_mode_default = False  # Single-pass for simplicity

        normal_message = "I had a good day today"
        result = generate_grounded_response(
            mock_provider,
            mock_cfg,
            user_message=normal_message,
            retrieved_context="",
        )

        # Should call provider for draft
        mock_provider.generate.assert_called_once()
        assert "This is a normal response" in result["answer"]

    def test_crisis_patterns_coverage(self):
        """Verify several crisis patterns trigger the gate."""
        mock_provider = MagicMock()
        mock_cfg = MagicMock()

        crisis_phrases = [
            "I want to end my life",
            "I'm going to hurt myself",
            "I'm suicidal",
            "I want to jump off a bridge",
        ]

        for phrase in crisis_phrases:
            result = generate_grounded_response(
                mock_provider,
                mock_cfg,
                user_message=phrase,
                retrieved_context="",
            )
            assert result["answer"] == CRISIS_SUPPORT_MESSAGE
            assert result["model"] == "crisis_gate"
