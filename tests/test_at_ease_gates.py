"""Tests for the two "feel at ease" gates.

1. The distress tier: elevated distress that is NOT crisis (hopelessness,
   worthlessness, self-blame) gets a steadying acknowledgement, while crisis
   still wins and ordinary entries stay untouched.
2. The assistant tone floor: nothing the assistant writes back may blame,
   dismiss, or diagnose the user.

Both are deterministic code gates rather than model judgment, for the same
reason _strip_ungrounded_quotes is: a small local verifier misses tone quietly,
and this text is read by someone who is often already struggling.
"""

from __future__ import annotations

import pytest

import app as app_module


# ── Distress detection ─────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "entry",
    [
        "I feel like a failure at everything I touch.",
        "I'm such a failure.",
        "I hate myself for saying yes again.",
        "What's the point of any of this anymore.",
        "I'm not good enough for this role.",
        "I feel worthless lately.",
        "I can't do anything right.",
        "Nothing I do matters.",
    ],
)
def test_self_directed_distress_is_detected(entry):
    assert app_module._is_distressed(entry) is True


@pytest.mark.parametrize(
    "entry",
    [
        "Busy week at work but I handled it.",
        "Grateful for a quiet morning and good coffee.",
        # "failure" about a thing, not the person.
        "The project is a failure but the team learned a lot.",
    ],
)
def test_ordinary_entries_are_not_flagged_as_distress(entry):
    assert app_module._is_distressed(entry) is False


@pytest.mark.parametrize(
    "entry",
    [
        "My friend said I am worthless, which hurt.",
        "He called me useless during the argument.",
        "She thinks I am not good enough, but I disagree.",
    ],
)
def test_reported_speech_is_not_treated_as_self_judgment(entry):
    """Someone else's cruelty is not the user judging themselves; answering it
    with "that sounds hard about yourself" would misread the entry."""
    assert app_module._is_distressed(entry) is False


def test_user_quoting_their_own_self_judgment_still_counts():
    """The reported-speech guard only excuses third-party subjects, so the user
    reporting their own words must still reach the distress tier."""
    assert app_module._is_distressed("I said I'm a failure and I meant it.") is True


# ── Gate precedence and effects ────────────────────────────────────────────
class TestGateTiers:
    def test_crisis_takes_precedence_over_distress(self):
        """A self-harm entry that also contains self-blame must get the support
        pathway, never the softer steadying framing."""
        entry = "I'm such a failure and I don't want to be here anymore."
        out = app_module._apply_reframe_gate(
            {"summary": "s", "reframe": "a cheerful reframe"}, entry, {}
        )
        assert out["crisis_support"] is True
        assert out["distress_support"] is False
        assert out["reframe"] == ""
        assert out["support_message"]

    def test_distress_adds_steadying_message_but_keeps_reframe(self):
        """Below crisis, a gentle reframe is appropriate, so it must survive."""
        out = app_module._apply_reframe_gate(
            {"summary": "s", "reframe": "a gentle reframe"}, "I feel like a failure.", {}
        )
        assert out["distress_support"] is True
        assert out["steadying_message"]
        assert out["reframe"] == "a gentle reframe"
        assert out["crisis_support"] is False

    def test_ordinary_entry_gets_neither_message(self):
        out = app_module._apply_reframe_gate(
            {"summary": "s", "reframe": "r"}, "Busy but productive day.", {}
        )
        assert out["crisis_support"] is False
        assert out["distress_support"] is False
        assert out["steadying_message"] == ""

    def test_steadying_message_is_not_medicalised(self):
        """This tier is ordinary human difficulty, not an emergency. Pointing it
        at crisis services would both misread it and alienate the user."""
        message = app_module.DISTRESS_STEADYING_MESSAGE.lower()
        for clinical in ("crisis line", "emergency", "hotline", "therapist", "doctor"):
            assert clinical not in message


# ── Assistant tone floor ───────────────────────────────────────────────────
class TestHarshOutputIsCaught:
    @pytest.mark.parametrize(
        "text",
        [
            "You brought this on yourself by never setting boundaries.",
            "You should just stop complaining.",
            "It's not that bad, others have it worse.",
            "You're being dramatic about a normal workload.",
            "Stop overthinking every detail.",
            "You have depression and should accept it.",
            "That is your own fault.",
        ],
    )
    def test_blaming_dismissive_or_diagnostic_text_is_flagged(self, text):
        assert app_module._find_harsh_content({"summary": text}) != []

    @pytest.mark.parametrize(
        "text",
        [
            "You are carrying a heavy workload and feeling stretched.",
            "You might try naming one deadline to renegotiate.",
            "Consider a short break before responding.",
            "Naming the feeling plainly, as you did, helps.",
        ],
    )
    def test_gentle_supportive_text_passes(self, text):
        """Ordinary advice must not trip the filter, or the analysis is gutted."""
        assert app_module._find_harsh_content({"summary": text}) == []

    def test_harsh_list_items_are_stripped_and_gentle_ones_kept(self):
        analysis = {
            "summary": "A hard week at work.",
            "coping_suggestions": [
                "You should just stop complaining and get on with it.",
                "Consider discussing workload with your manager.",
            ],
            "journaling_feedback": ["Stop overthinking every detail."],
            "patterns": ["saying yes when already at capacity"],
            "triggers": [],
        }
        out = app_module._strip_harsh_items(analysis)
        assert out["coping_suggestions"] == ["Consider discussing workload with your manager."]
        assert out["journaling_feedback"] == []
        assert out["patterns"] == ["saying yes when already at capacity"]

    def test_crisis_verdict_from_retrieved_history_is_the_verifiers_job_not_the_gates(self):
        """Regression guard for a live defect: because every entry is written to
        the RAG store, a later entry retrieves the earlier one as context, and the
        verifier was judging crisis from that retrieved history. One bad night
        then marked every subsequent entry as crisis, so "had a good day" was
        answered with emergency resources.

        The fix is in VERIFIER_SYSTEM_PROMPT (judge crisis from CURRENT_ENTRY
        only). The gate itself is unchanged and must stay fail-closed, so this
        asserts the instruction is present rather than re-testing the gate: the
        behavioural check needs a live model and lives in the manual run.
        """
        from verifier_prompts import VERIFIER_SYSTEM_PROMPT

        prompt = VERIFIER_SYSTEM_PROMPT.upper()
        assert "CURRENT_ENTRY ONLY" in prompt
        assert "NEVER DRIVE CRISIS_DETECTED" in prompt

    def test_verifier_prompt_separates_self_criticism_from_crisis(self):
        """The verifier was reading "I'm not good enough" as suicidal ideation,
        which shadowed the distress tier and pushed hotline messaging at someone
        disappointed about a promotion."""
        from verifier_prompts import VERIFIER_SYSTEM_PROMPT

        prompt = VERIFIER_SYSTEM_PROMPT.lower()
        assert "not the same as harsh self-criticism" in prompt
        assert "i'm not good enough" in prompt

    def test_user_quotes_are_never_stripped(self):
        """quotes_from_user echoes the user's own words. Harsh phrasing there is
        theirs, and censoring it would edit the person's own journal."""
        analysis = {
            "summary": "ok",
            "quotes_from_user": ["it's my own fault", "I'm being dramatic"],
            "coping_suggestions": [],
        }
        out = app_module._strip_harsh_items(dict(analysis))
        assert out["quotes_from_user"] == analysis["quotes_from_user"]
        assert app_module._find_harsh_content(analysis) == []
