"""Tests for valence scoring and valence-aware reordering.

Focused on the failure modes that actually bit during development (recorded in
docs/IMPROVEMENTS.md section 9), not on happy-path scoring:

  1. Mixed-valence text collapses to neutral under a scalar score. This is the
     normal case for emotionally significant journal entries, not an edge case.
  2. A neutral or mixed entry must never be demoted. Demoting it regressed the
     temporal retrieval category from 1.000 to 0.500.
  3. Agreement must never promote, or valence overrides relevance entirely.
"""

from __future__ import annotations

import pytest

import valence


class TestScoring:
    def test_clear_positive_and_negative(self):
        assert valence.classify("Grateful and calm today, felt close to them") == "positive"
        assert valence.classify("Exhausted and anxious, dreading tomorrow") == "negative"

    def test_negation_flips_polarity(self):
        """'not good' must not score as positive just because 'good' appears."""
        assert valence.score("this was not good at all") < 0
        assert valence.classify("I am not happy about this") == "negative"

    def test_empty_and_unrecognised_text_is_neutral(self):
        assert valence.score("") == 0.0
        assert valence.classify("the meeting is at four on tuesday") == "neutral"

    def test_mixed_valence_collapses_to_neutral(self):
        """The known structural limit of a scalar score.

        Both of these are emotionally rich entries whose positive and negative
        terms cancel. Recorded as a limitation rather than fixed, because fixing
        it needs a two-dimensional score, not a better word list.
        """
        assert valence.classify("Excited about the new city but grieving this one") == "neutral"
        assert valence.classify(
            "The presentation actually went fine. All that dread for nothing."
        ) == "neutral"


class TestAgreementContract:
    def test_neutral_never_agrees_and_never_opposes(self):
        """Neutral means 'no evidence', which is distinct from disagreement.

        Conflating the two is what caused the temporal regression.
        """
        neutral = "the meeting is at four on tuesday"
        positive = "grateful and calm today"
        assert not valence.agrees(positive, neutral)
        assert not valence.opposes(positive, neutral)

    def test_opposes_requires_both_sides_non_neutral(self):
        assert valence.opposes("grateful and calm today", "exhausted and hopeless")
        assert not valence.opposes("grateful and calm today", "")


class TestPartitionByAgreement:
    POSITIVE_QUERY = "a good day where I felt appreciated and grateful"

    def test_neutral_query_leaves_order_untouched(self):
        texts = ["exhausted and hopeless", "grateful and calm"]
        assert valence.partition_by_agreement("the meeting is on tuesday", texts) == [0, 1]

    def test_opposing_entry_is_demoted(self):
        texts = ["exhausted, hopeless and worthless", "grateful and calm today"]
        assert valence.partition_by_agreement(self.POSITIVE_QUERY, texts) == [1, 0]

    def test_neutral_entry_is_not_demoted_below_an_agreeing_one(self):
        """Regression guard for the measured temporal failure.

        A neutral entry the ranker placed first must stay first. An earlier
        promoting variant moved it behind the agreeing entry at index 1, which
        dropped the target out of top-k on the real corpus.
        """
        texts = ["the meeting is at four on tuesday", "grateful and calm today"]
        assert valence.partition_by_agreement(self.POSITIVE_QUERY, texts) == [0, 1]

    def test_agreement_alone_never_promotes(self):
        """Relevance is primary; valence only demotes contradictions.

        Neither entry opposes the query, so the ranker's order must survive
        even though the second one agrees and the first does not.
        """
        texts = ["the meeting is at four on tuesday", "grateful, calm and appreciated"]
        assert valence.partition_by_agreement(self.POSITIVE_QUERY, texts) == [0, 1]

    def test_empty_candidate_list(self):
        assert valence.partition_by_agreement(self.POSITIVE_QUERY, []) == []

    def test_reordering_never_drops_or_duplicates_candidates(self):
        """Whatever the valence verdict, the candidate set must be preserved.

        A reranker that silently loses a candidate is worse than one that
        ranks badly.
        """
        texts = [
            "exhausted and hopeless",
            "the meeting is at four",
            "grateful and calm",
            "",
        ]
        order = valence.partition_by_agreement(self.POSITIVE_QUERY, texts)
        assert sorted(order) == list(range(len(texts)))


@pytest.mark.parametrize(
    "text",
    ["I feel like a failure", "I hate myself", "nothing matters anymore"],
)
def test_distress_language_scores_negative(text):
    """Cross-check against the distress tier in app.py.

    These are the phrasings _DISTRESS_PATTERNS exists to catch. Valence must
    agree with that classification, or retrieval could ground a distressed
    entry in the user's cheerful ones.
    """
    assert valence.classify(text) == "negative"


class TestGratitudeCoverage:
    """Regression guard for a gap found by independent validation.

    evals/valence_external_validation.py (scored against GoEmotions, a
    third-party human-annotated dataset) found "Thank you for asking
    questions..." and "100%! Congrats on your job too!" both scoring neutral --
    the lexicon had "thankful"/"thanks" but not the bare word "thank", and no
    congratulations word at all. Fixed as general lexicon coverage, not by
    matching those two sentences; these tests check the general words, not the
    exact failing text.
    """

    def test_bare_thank_scores_positive(self):
        assert valence.classify("Thank you so much for the help") == "positive"

    def test_congrats_scores_positive(self):
        assert valence.classify("Congrats on the new job!") == "positive"

    def test_congratulations_scores_positive(self):
        assert valence.classify("Congratulations, that's wonderful news") == "positive"


class TestEmoticons:
    """Regression guard for the second gap found by the same validation run.

    "I'm really sorry about your situation :(" scored positive: the tokenizer
    (`_TOKEN = re.compile(r"[a-z']+")`) strips emoticons before scoring ever
    sees them, so a clear, visible sentiment signal was invisible by
    construction.
    """

    def test_negative_emoticon_flips_a_neutral_sentence(self):
        assert valence.classify("That was rough today :(") == "negative"

    def test_positive_emoticon_flips_a_neutral_sentence(self):
        assert valence.classify("That was great today :)") == "positive"

    def test_emoticon_with_trailing_punctuation_still_matches(self):
        """The boundary regex must not require whitespace immediately after."""
        assert valence.classify("So relieved it's over :).") == "positive"

    def test_doubled_emoticon_for_emphasis_still_matches(self):
        assert valence.classify("Everyone left :((") == "negative"

    def test_code_like_text_does_not_false_positive(self):
        """'):' and '():' patterns show up in code and list markers, not sadness."""
        assert valence.classify("def foo(): return") == "neutral"
        assert valence.classify("a) option one b) option two") == "neutral"
