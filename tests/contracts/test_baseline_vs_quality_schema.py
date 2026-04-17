"""
Baseline vs Quality contract tests (DPO starvation canary).

The DPO preference-pair pipeline relies on baseline_json mode being
*deliberately weaker* than quality mode. If someone later "improves"
the baseline by tightening its prompt or raising its retry count,
delta scores collapse to zero and the preference dataset starves.

These tests pin the weakness contract so that regression is loud.

Today the baseline knobs live inline inside ``app._run_baseline`` as
constants (WEAKER_SYSTEM_PROMPT, temperature, max_retries). When those
are hoisted to ``config.py`` or a dataclass (see upgrade 06 risks and
upgrade 07 cleanup), the source-inspection approach here should be
replaced with direct attribute assertions.

These tests must run without Ollama: we only inspect source / imports,
we do not call the pipeline.
"""

from __future__ import annotations

import inspect
import re

from generator_prompts import DRAFT_SYSTEM_PROMPT
from schemas.analysis import AnalysisOutput


def _baseline_source() -> str:
    # Import here so app.py import happens at test-collection time only
    # when this test actually runs (keeps collection cheap).
    import app  # noqa: WPS433

    return inspect.getsource(app._run_baseline)


def _quality_source() -> str:
    import app  # noqa: WPS433

    return inspect.getsource(app._run_quality_pipeline)


class TestBaselineIsWeakerThanQuality:
    def test_baseline_has_distinct_weaker_system_prompt(self):
        """
        Baseline must use a system prompt that is not DRAFT_SYSTEM_PROMPT.
        If baseline is ever changed to import DRAFT_SYSTEM_PROMPT directly,
        this test fires.
        """
        src = _baseline_source()
        # Baseline defines its own WEAKER_SYSTEM_PROMPT local.
        assert "WEAKER_SYSTEM_PROMPT" in src, (
            "Baseline path no longer defines WEAKER_SYSTEM_PROMPT. "
            "If this constant was renamed or hoisted, update this test to "
            "read it from its new location rather than loosening the assertion."
        )
        # And does not delegate to the strong DRAFT_SYSTEM_PROMPT.
        assert "DRAFT_SYSTEM_PROMPT" not in src, (
            "Baseline pipeline appears to reference DRAFT_SYSTEM_PROMPT. "
            "Doing so erases the baseline/quality delta and will starve "
            "the DPO preference dataset."
        )

    def test_baseline_temperature_is_nonzero(self):
        """
        Baseline runs at a warmer sampling temperature than quality
        (which runs at the llm_client default of 0.0). 0.3 is the
        current pinned value.
        """
        src = _baseline_source()
        m = re.search(r"temperature\s*=\s*([0-9]*\.?[0-9]+)", src)
        assert m is not None, "Baseline no longer passes an explicit temperature."
        temp = float(m.group(1))
        assert temp > 0.0, (
            f"Baseline temperature {temp} is not warmer than quality's 0.0 default. "
            "Baseline must stay stochastic enough to produce weaker drafts; "
            "collapsing it to 0.0 will shrink DPO preference-pair deltas."
        )

    def test_baseline_max_retries_is_smaller_than_quality(self):
        """
        Baseline is documented as using 3 retries; quality uses 5.
        This test asserts the relationship (baseline < quality) rather
        than the exact number so a future bump to 4/6 doesn't break it.
        """
        b_src = _baseline_source()
        q_src = _quality_source()

        b_retries = [int(x) for x in re.findall(r"max_retries\s*=\s*(\d+)", b_src)]
        q_retries = [int(x) for x in re.findall(r"max_retries\s*=\s*(\d+)", q_src)]

        assert b_retries, "Baseline no longer passes an explicit max_retries."
        assert q_retries, "Quality no longer passes an explicit max_retries."

        assert max(b_retries) < max(q_retries), (
            f"Baseline max_retries={max(b_retries)} is not strictly less than "
            f"quality max_retries={max(q_retries)}. Tightening baseline "
            "will reduce parse failures on purpose and collapse the "
            "quality-vs-baseline delta the DPO pipeline depends on."
        )


class TestModeOutputSchema:
    """
    Both modes must produce AnalysisOutput-valid JSON so downstream
    tooling (DPO pair builder, evaluator, insight formatter) does not
    have to branch on mode.
    """

    def test_sample_valid_payload_passes_analysis_schema(
        self, valid_analysis_json
    ):
        # Sanity: the fixture itself is a valid AnalysisOutput.
        AnalysisOutput.model_validate(valid_analysis_json)
