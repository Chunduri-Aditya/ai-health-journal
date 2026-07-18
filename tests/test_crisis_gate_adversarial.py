"""Adversarial regression tests for the crisis reframe gate.

Written from the aggressive-audit findings report. Two kinds of test here:

1. Real regressions (no xfail): behavior the gate DOES correctly hold today.
   These must keep passing.
2. Documented gaps (xfail, strict=True): behavior the gate does NOT currently
   catch, proven by direct execution against _is_crisis. Each is real-world
   crisis phrasing that bypasses the regex floor when the verifier also misses
   or fails. strict=True means an unexpected pass (XPASS) fails the suite, so
   a future fix is caught here rather than silently going unnoticed.

Scope note: these test the DETERMINISTIC FLOOR only (_CRISIS_PATTERNS / safety
flag substring match). The verifier LLM is the primary defense and is expected
to catch most of these via crisis_detected; the floor's job is specifically to
catch what the verifier misses. That's exactly the scenario simulated here:
clean_verdict / verifier-failure verdict, i.e. the floor operating alone.
"""

from pathlib import Path

import pytest

from app import _apply_reframe_gate, _is_crisis

_TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "index.html"

_CLEAN_VERDICT = {"crisis_detected": False, "safety_flags": []}


def _fresh_verdict() -> dict:
    return dict(_CLEAN_VERDICT)


# ── Held: literal phrasing the floor correctly catches ─────────────────────
class TestFloorCatchesLiteralPhrasing:
    @pytest.mark.parametrize(
        "entry",
        [
            "I want to kill myself tonight",
            "I've been feeling suicidal all week",
            "I've been engaging in selfharm again",
            "I don't want to wake up tomorrow",
            "I've decided to take my own life",
        ],
    )
    def test_literal_self_harm_phrasing_caught(self, entry):
        assert _is_crisis(entry, _fresh_verdict()) is True

    def test_mixed_positive_and_crisis_signal_still_caught(self):
        entry = "Had a great day at work! Unrelated: I want to kill myself."
        assert _is_crisis(entry, _fresh_verdict()) is True

    def test_crisis_phrase_buried_in_long_entry_still_caught(self):
        entry = ("Today was busy. " * 50) + "I want to end my life."
        assert _is_crisis(entry, _fresh_verdict()) is True

    def test_hyphenated_self_harm_flag_from_verifier_caught(self):
        verdict = {"crisis_detected": False, "safety_flags": ["self-harm risk detected"]}
        assert _is_crisis("I feel really down today.", verdict) is True

    # FIXED (was TestFloorGapsInRealWorldPhrasing, xfail): the regex floor now
    # covers these common non-literal crisis phrasings. Verified against a
    # false-positive stress test (idiom coverage below, plus "overdose of
    # nostalgia" / AWS "KMS" collisions caught during that testing and fixed
    # by narrowing the pattern before landing this) before moving these out
    # of xfail.
    @pytest.mark.parametrize(
        "entry",
        [
            "I just want to end it all",
            "I no longer want to exist",
            "I don't want to be alive anymore",
            "I want to unalive myself",
            "everyone would be better off if I was dead",
            "I think I'm just going to give up on life",
            "I've been thinking about taking all my pills at once",
            "I keep thinking about jumping off the bridge near my house",
        ],
    )
    def test_realworld_phrasing_now_caught_by_floor(self, entry):
        assert _is_crisis(entry, _fresh_verdict()) is True


class TestFloorDoesNotFalsePositiveOnIdioms:
    @pytest.mark.parametrize(
        "entry",
        [
            "this commute is killing me and work is exhausting",
            "I was literally dying laughing at that meme",
            "this job is killing me, so stressful",
            # Regression coverage for false positives found while expanding
            # the floor: "kms" collides with the AWS acronym, bare "overdose"
            # collides with the common idiom "an overdose of X" -- both were
            # narrowed/dropped in app.py before landing.
            "I overdosed on caffeine today, way too much coffee",
            "This show gives me a serious overdose of nostalgia",
            "The AWS KMS key rotation finally finished",
            "kms is such a pain to configure in our infra",
            "I gave up on my diet, no willpower today",
            "I gave up on that project, it wasn't working out",
        ],
    )
    def test_idiom_not_flagged(self, entry):
        assert _is_crisis(entry, _fresh_verdict()) is False

    def test_bungee_jumping_off_a_bridge_is_an_accepted_false_positive(self):
        """Known, deliberate tradeoff (see app.py comment above
        _CRISIS_PATTERNS): "jumping off a bridge" also matches benign
        recreational entries. Kept anyway -- catching method-specific
        suicidal ideation outweighs an occasional unnecessary supportive
        message. This test documents the tradeoff is intentional, not an
        oversight.
        """
        entry = "Watched people bungee jumping off a bridge on vacation, terrifying"
        assert _is_crisis(entry, _fresh_verdict()) is True


# ── Gap: still not covered, and why ─────────────────────────────────────────
@pytest.mark.xfail(
    reason="intentionally NOT covered: too easily a benign statement (moving, "
    "changing jobs, retiring) to regex safely without a much higher false-"
    "positive rate; left to the verifier's judgment",
    strict=True,
)
def test_vague_departure_phrasing_not_caught_by_floor():
    assert _is_crisis("I won't be here much longer, I've decided.", _fresh_verdict()) is True


@pytest.mark.xfail(reason="regex floor gap: non-English self-harm phrasing", strict=True)
def test_non_english_self_harm_phrasing_caught_by_floor():
    assert _is_crisis("quiero matarme esta noche", _fresh_verdict()) is True


# FIXED (was xfail): safety_flags are now normalized (underscore/hyphen ->
# space) before matching against a broadened term list, so paraphrases and
# snake_case flags from the verifier are caught, not just the 4 original
# hardcoded exact substrings.
@pytest.mark.parametrize(
    "flags",
    [
        ["self_harm_ideation"],
        ["risk of self injury"],
        ["harm ideation present"],
        ["possible danger to self"],
    ],
)
def test_paraphrased_safety_flag_now_caught(flags):
    verdict = {"crisis_detected": False, "safety_flags": flags}
    assert _is_crisis("I feel really down today.", verdict) is True


class TestHarmToOthersFloor:
    """FIXED (was test_threat_to_others_caught_by_floor, xfail):
    _HARM_TO_OTHERS_PATTERNS adds a deterministic floor for explicit,
    first-person, planned intent to harm someone else. Deliberately much
    narrower than the self-harm floor -- "I'm going to kill him" is common,
    harmless hyperbole in a way "I want to kill myself" essentially never
    is, so this requires an explicit intent marker (going to / planning to /
    I'll) directly attached to the harm clause. See app.py for the full
    false-positive stress test this was designed against.
    """

    def test_explicit_threat_to_others_caught(self):
        verdict = {"crisis_detected": False, "safety_flags": []}
        entry = "I'm going to make them pay, I'll hurt him badly."
        assert _is_crisis(entry, verdict) is True

    @pytest.mark.parametrize(
        "entry",
        [
            "I'm planning to hurt him this weekend.",
            "I am going to attack them tomorrow.",
        ],
    )
    def test_planned_intent_variants_caught(self, entry):
        assert _is_crisis(entry, _fresh_verdict()) is True

    @pytest.mark.parametrize(
        "entry",
        [
            "my mom is going to kill me for this",
            "I could kill him right now, so annoying",
            "ugh he ate my leftovers, I'm going to kill him",
            "we're going to destroy them in the game tonight",
            "I'll crush him in chess",
            "my sister is going to make me pay for this prank",
            "that movie hurt him deeply, he cried the whole time",
            "the breakup really hurt her badly",
            "watching him struggle really hurt me badly",
            "he said something that hurt her badly last week",
        ],
    )
    def test_common_hyperbole_not_flagged(self, entry):
        """The precision-critical regression: bare "kill him" and third-
        person emotional "hurt ___ badly" are both extremely common non-
        threatening phrasing and must not fire the floor.
        """
        assert _is_crisis(entry, _fresh_verdict()) is False


# ── Worst case: verifier fails AND regex misses -> positivity survives ─────
class TestWorstCaseVerifierFailure:
    def test_verifier_exception_fallback_plus_regex_gap_leaks_reframe(self):
        """Reproduces the exact fallback verdict shape _run_quality_pipeline
        builds when the verifier call raises (app.py, except branch around
        line 391) — no crisis_detected key at all — combined with phrasing the
        regex floor does not catch. Confirms the reframe currently survives.
        """
        verifier_failed_verdict = {
            "groundedness_score": 0.5,
            "unsupported_claims": [],
            "safety_flags": [],
            "rewrite_required": False,
            "rewrite_instructions": "",
        }
        entry = "I've decided I won't be here much longer."
        draft = {
            "reframe": "Every day is a fresh chance to find your footing again!",
            "crisis_support": False,
            "support_message": "",
        }
        out = _apply_reframe_gate(dict(draft), entry, verifier_failed_verdict)
        # Documents CURRENT (unsafe) behavior so a fix is visible as a diff here,
        # not silently. This is intentionally NOT xfail: it is the concrete,
        # reproducible worst case and should be the first thing a fix flips.
        assert out["crisis_support"] is False
        assert out["reframe"] != ""


# ── Operational misconfiguration: blank crisis message env var ─────────────
def test_blank_crisis_message_env_var_falls_back_to_default(monkeypatch):
    """FIXED: CRISIS_SUPPORT_MESSAGE = os.getenv("AIHJ_CRISIS_MESSAGE") or
    (...) now falls back to the built-in default even when the env var is
    explicitly set to an empty string, not just when it's unset entirely
    (os.getenv's default= parameter only ever applied to the unset case).
    The constant is computed once at module import time, so verifying the
    actual fix means reloading the module under the patched environment --
    reloaded again in finally to restore normal state for later tests.
    """
    import importlib

    import app as app_module

    monkeypatch.setenv("AIHJ_CRISIS_MESSAGE", "")
    try:
        importlib.reload(app_module)
        assert app_module.CRISIS_SUPPORT_MESSAGE != ""
        assert "crisis line" in app_module.CRISIS_SUPPORT_MESSAGE.lower()
    finally:
        importlib.reload(app_module)


def test_apply_reframe_gate_has_no_independent_defense_for_blank_message():
    """Defense-in-depth note, not a bug to fix: _apply_reframe_gate itself
    trusts CRISIS_SUPPORT_MESSAGE as given and has no fallback of its own --
    the ONLY thing preventing a silent crisis render is the env-var-parsing
    fix above. Documents that this is a single point of correctness, in case
    a future refactor ever bypasses the module-level constant.
    """
    from app import _apply_reframe_gate

    draft = {"reframe": "Stay positive!", "crisis_support": False, "support_message": ""}
    verdict = {"crisis_detected": True, "safety_flags": []}
    out = _apply_reframe_gate(dict(draft), "I want to kill myself.", verdict)
    # CRISIS_SUPPORT_MESSAGE is non-empty (verified above), so this real
    # module state IS correctly non-silent -- confirming the gate uses it.
    from app import CRISIS_SUPPORT_MESSAGE

    assert out["support_message"] == CRISIS_SUPPORT_MESSAGE
    assert out["support_message"] != ""


# ── Frontend: does the crisis output actually reach the browser? ───────────
class TestFrontendRendersCrisisOutput:
    """The backend can compute crisis_support/support_message correctly, and
    the browser must actually show it. templates/index.html's
    renderAnalysisCards() builds the DOM shown in the insight box from
    data.analysis, and REPLACES the plain-text data.insight entirely
    (insightBox.innerHTML = ""; insightBox.appendChild(cards)) whenever
    data.analysis is present -- which it always is in quality mode. If
    renderAnalysisCards() does not read support_message/crisis_support/
    reframe/journaling_feedback, none of this session's safety or coaching
    work ever reaches the user, regardless of whether the Python gate fires
    correctly.

    FIXED (was CRITICAL, confirmed by grep showing zero occurrences of all
    four field names in templates/index.html; now renders crisis_support/
    support_message in a dedicated .crisis-support-box, reframe in
    .reframe-box -- mutually exclusive with the crisis box, mirroring
    app._format_insight -- and journaling_feedback as a "Journaling tips"
    section reusing the existing suggestions-list style).

    This is a text-search regression, not a JS unit test (no JS runner in
    this repo) -- it exists so a regression must touch the fields checked
    here to go unnoticed.
    """

    def test_render_function_references_all_four_new_fields(self):
        assert _TEMPLATE.exists(), f"template not found: {_TEMPLATE}"
        source = _TEMPLATE.read_text(encoding="utf-8")
        # Isolate renderAnalysisCards() (used for both the live submit path
        # and the history detail view) so a match elsewhere in the file
        # doesn't hide the gap.
        start = source.index("function renderAnalysisCards")
        end = source.index("\n    function ", start + 1)
        render_fn_source = source[start:end]

        missing = [
            field
            for field in ("support_message", "crisis_support", "reframe", "journaling_feedback")
            if field not in render_fn_source
        ]
        assert not missing, (
            f"renderAnalysisCards() in templates/index.html does not render: "
            f"{missing}. The backend crisis gate can fire correctly and the "
            f"browser will still show nothing for it."
        )
