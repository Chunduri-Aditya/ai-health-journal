"""Tests for the reframe-quality rubric.

Mirrors the structure of tests/test_valence.py: the rubric's job is to catch
specific harmful patterns, so the tests target one pattern at a time rather than
scoring realistic-looking text end to end. The validation harness
(evals/reframe_quality_eval.py) covers end-to-end separation against a labeled
set; these are the per-detector unit boundaries.
"""

from __future__ import annotations

import pytest

from evals.reframe_quality import GROUNDING_MIN_OVERLAP, score_reframe

ENTRY = "Got passed over for the promotion again after two years on the team."


class TestEmptyReframe:
    def test_empty_reframe_is_acceptable(self):
        """The crisis gate clears reframe on purpose; that must never be flagged."""
        score = score_reframe(ENTRY, "")
        assert score.acceptable
        assert not score.offenders

    def test_none_like_missing_field_does_not_crash(self):
        score = score_reframe(ENTRY, "   ")
        assert score.acceptable


class TestMinimising:
    @pytest.mark.parametrize(
        "phrase",
        ["at least you still have a job", "it could be worse", "it's not that bad"],
    )
    def test_minimising_phrases_are_flagged(self, phrase):
        score = score_reframe(ENTRY, f"That's rough, but {phrase}.")
        assert score.minimising
        assert not score.acceptable

    def test_at_least_inside_an_otherwise_good_reframe_still_flags(self):
        """One bad phrase disqualifies regardless of surrounding quality.

        Failure modes are not tradeable against a good average (see
        ReframeScore.acceptable). A single minimising clause is still harmful
        even wrapped in validating language.
        """
        text = (
            "That sounds genuinely hard and it makes sense you'd feel this way. "
            "At least you have the experience to show for it."
        )
        score = score_reframe(ENTRY, text)
        assert score.minimising
        assert not score.acceptable


class TestToxicPositivity:
    @pytest.mark.parametrize(
        "phrase",
        ["everything happens for a reason", "look on the bright side", "cheer up"],
    )
    def test_toxic_positivity_phrases_are_flagged(self, phrase):
        score = score_reframe(ENTRY, f"I know it's tough, but {phrase}.")
        assert score.toxic_positivity
        assert not score.acceptable


class TestCommanding:
    def test_you_must_is_flagged(self):
        score = score_reframe(ENTRY, "You must talk to your manager about this tomorrow.")
        assert score.commanding

    def test_you_should_is_flagged(self):
        score = score_reframe(ENTRY, "You should stop dwelling on it and move forward.")
        assert score.commanding

    def test_invitational_hedge_is_not_commanding(self):
        """The product's intended behaviour: offering, not ordering.

        generator_prompts.py rule 12 requires suggestions to read as invitations
        ("you might try") rather than orders ("you must"). This is the case the
        detector must NOT flag, or the rubric would reject correct behavior.
        """
        score = score_reframe(
            ENTRY, "It might help to talk to your manager, if that feels right to you."
        )
        assert not score.commanding


class TestGeneric:
    def test_unrelated_platitude_with_no_grounding_is_flagged(self):
        score = score_reframe(ENTRY, "Growth often follows discomfort, and this will pass.")
        assert score.generic

    def test_grounded_text_is_not_generic_even_if_short(self):
        score = score_reframe(ENTRY, "Two years on that team is a long stretch of effort.")
        assert not score.generic

    def test_low_overlap_with_validation_language_is_not_generic(self):
        """Generic requires BOTH low overlap AND no acknowledgement.

        A short but validating reframe ("that sounds really hard") can
        legitimately have low word overlap with the entry; it is not thereby
        ungrounded nonsense, it is brief.
        """
        score = score_reframe(ENTRY, "That sounds genuinely hard.")
        assert not score.generic


class TestInvalidating:
    def test_pivot_with_no_acknowledgement_is_flagged(self):
        """Grounded but dismissive: mentions the entry, still skips acknowledgement.

        Text with zero grounding and a pivot phrase is classified as `generic`
        instead (see the `and not result.generic` guard in reframe_quality.py):
        a statement unmoored from the entry is a different failure than one that
        references the entry and still dismisses it. This case is written with
        the team/promotion detail present so it exercises `invalidating`
        specifically rather than falling through to `generic`.
        """
        score = score_reframe(
            ENTRY,
            "The good news is the promotion and the team situation won't matter "
            "in a few years.",
        )
        assert score.invalidating
        assert not score.generic

    def test_pivot_with_prior_acknowledgement_is_not_invalidating(self):
        """The same pivot phrase is fine once the feeling has been named first.

        Distinguishes "acknowledge then reframe" (the intended shape) from
        "skip straight to reassurance" (the harmful one).
        """
        score = score_reframe(
            ENTRY,
            "That's a genuinely difficult thing to sit with. The good news is "
            "this is one decision, not a verdict on your work.",
        )
        assert not score.invalidating


class TestAcceptableIsNotAverage:
    def test_a_single_failure_mode_disqualifies_regardless_of_others(self):
        """acceptable is a veto, not a weighted score.

        A reframe that is well-grounded and validating everywhere else is still
        unacceptable if it also commands. Verified directly against the
        dataclass rather than through text, so this test does not depend on any
        particular regex continuing to fire on any particular phrase.
        """
        from evals.reframe_quality import ReframeScore

        score = ReframeScore(commanding=True)
        assert not score.acceptable


def test_grounding_overlap_is_reproducible_and_bounded():
    score = score_reframe(ENTRY, "Two years on that team, promotion, joined in March.")
    assert 0.0 <= score.grounding_overlap <= 1.0
    assert score.grounding_overlap > GROUNDING_MIN_OVERLAP
