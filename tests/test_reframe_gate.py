"""Deterministic tests for the crisis reframe gate.

The gate is the safety-critical seam of the reframe feature: it must never let
the positivity/reframe path fire on a self-harm entry, and must fail closed when
the verifier misses or errors. These tests exercise that boundary with no LLM.
"""

from app import CRISIS_SUPPORT_MESSAGE, _apply_reframe_gate, _is_crisis


def _draft(reframe: str = "You did your best today.") -> dict:
    return {
        "summary": "s",
        "emotions": [],
        "patterns": [],
        "triggers": [],
        "coping_suggestions": [],
        "quotes_from_user": [],
        "confidence": 0.5,
        "journaling_feedback": [],
        "reframe": reframe,
        "crisis_support": False,
        "support_message": "",
    }


def _verdict(*, crisis: bool = False, flags: list[str] | None = None) -> dict:
    return {
        "groundedness_score": 1.0,
        "unsupported_claims": [],
        "safety_flags": flags or [],
        "crisis_detected": crisis,
        "rewrite_required": False,
        "rewrite_instructions": "",
    }


class TestReframeGate:
    def test_ordinary_negative_keeps_reframe(self):
        out = _apply_reframe_gate(
            _draft("Try to be kind to yourself here."),
            "I failed my interview and feel worthless.",
            _verdict(),
        )
        assert out["reframe"] == "Try to be kind to yourself here."
        assert out["crisis_support"] is False
        assert out["support_message"] == ""

    def test_verifier_crisis_suppresses_reframe(self):
        out = _apply_reframe_gate(
            _draft("Everything will be great!"),
            "Work was stressful today.",
            _verdict(crisis=True),
        )
        assert out["reframe"] == ""
        assert out["crisis_support"] is True
        assert out["support_message"] == CRISIS_SUPPORT_MESSAGE

    def test_safety_flag_crisis_suppresses_reframe(self):
        out = _apply_reframe_gate(
            _draft("Cheer up, tomorrow is a new day!"),
            "a hard day",
            _verdict(flags=["possible self-harm ideation"]),
        )
        assert out["crisis_support"] is True
        assert out["reframe"] == ""

    def test_regex_floor_fires_when_verifier_misses(self):
        # Verifier missed it (crisis_detected False, no flags); the floor must catch it.
        out = _apply_reframe_gate(
            _draft("You've got this!"),
            "I want to kill myself tonight.",
            _verdict(),
        )
        assert out["crisis_support"] is True
        assert out["reframe"] == ""

    def test_idiom_does_not_falsely_trigger(self):
        assert _is_crisis("this commute is killing me and work is exhausting", _verdict()) is False

    def test_reflexive_phrases_detected(self):
        for entry in (
            "I want to die",
            "I don't want to live anymore",
            "I keep thinking about hurting myself",
            "I've been feeling suicidal",
            "sometimes I just want to end my life",
        ):
            assert _is_crisis(entry, _verdict()) is True, entry
