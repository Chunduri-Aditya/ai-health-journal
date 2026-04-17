"""
AnalysisOutput and VerifierVerdict edge-case tests.

These pin the post-parse validation contract for upgrade 06a:
- No silent empty outputs.
- No unknown fields slip through.
- Bounds on confidence / groundedness_score are enforced.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.analysis import AnalysisOutput
from schemas.verifier import VerifierVerdict


class TestAnalysisOutput:
    def test_happy_path(self, valid_analysis_json):
        out = AnalysisOutput.model_validate(valid_analysis_json)
        assert out.summary
        assert out.confidence == pytest.approx(0.7)

    def test_empty_object_rejected(self):
        with pytest.raises(ValidationError):
            AnalysisOutput.model_validate({})

    def test_missing_required_summary(self, valid_analysis_json):
        del valid_analysis_json["summary"]
        with pytest.raises(ValidationError):
            AnalysisOutput.model_validate(valid_analysis_json)

    def test_empty_summary_string_rejected(self, valid_analysis_json):
        valid_analysis_json["summary"] = ""
        with pytest.raises(ValidationError):
            AnalysisOutput.model_validate(valid_analysis_json)

    def test_confidence_out_of_bounds(self, valid_analysis_json):
        valid_analysis_json["confidence"] = 1.5
        with pytest.raises(ValidationError):
            AnalysisOutput.model_validate(valid_analysis_json)

        valid_analysis_json["confidence"] = -0.1
        with pytest.raises(ValidationError):
            AnalysisOutput.model_validate(valid_analysis_json)

    def test_extra_field_forbidden(self, valid_analysis_json):
        valid_analysis_json["invented_key"] = "should not pass"
        with pytest.raises(ValidationError):
            AnalysisOutput.model_validate(valid_analysis_json)

    def test_coping_suggestions_cap(self, valid_analysis_json):
        valid_analysis_json["coping_suggestions"] = [f"tip {i}" for i in range(11)]
        with pytest.raises(ValidationError):
            AnalysisOutput.model_validate(valid_analysis_json)

    def test_quotes_from_user_cap(self, valid_analysis_json):
        valid_analysis_json["quotes_from_user"] = [f"q{i}" for i in range(6)]
        with pytest.raises(ValidationError):
            AnalysisOutput.model_validate(valid_analysis_json)

    def test_optional_grounding_fields_default_empty(self, valid_analysis_json):
        out = AnalysisOutput.model_validate(valid_analysis_json)
        assert out.grounding_evidence == []
        assert out.grounding_sources == []
        assert out.uncertainties == []
        assert out.grounding_mode is None
        assert out.retrieval_top_k is None

    def test_mutable_default_is_not_shared_between_instances(self, valid_analysis_json):
        """Regression guard: shared mutable defaults would leak across instances."""
        a = AnalysisOutput.model_validate(valid_analysis_json)
        b = AnalysisOutput.model_validate(valid_analysis_json)
        a.emotions.append("loneliness")
        assert "loneliness" not in b.emotions


class TestVerifierVerdict:
    def test_happy_path(self, valid_verifier_json):
        v = VerifierVerdict.model_validate(valid_verifier_json)
        assert v.groundedness_score == pytest.approx(0.88)
        assert v.rewrite_required is False

    def test_empty_object_rejected(self):
        with pytest.raises(ValidationError):
            VerifierVerdict.model_validate({})

    def test_groundedness_out_of_bounds(self, valid_verifier_json):
        valid_verifier_json["groundedness_score"] = 1.1
        with pytest.raises(ValidationError):
            VerifierVerdict.model_validate(valid_verifier_json)

    def test_missing_rewrite_required(self, valid_verifier_json):
        del valid_verifier_json["rewrite_required"]
        with pytest.raises(ValidationError):
            VerifierVerdict.model_validate(valid_verifier_json)

    def test_rewrite_instructions_default_empty(self, valid_verifier_json):
        del valid_verifier_json["rewrite_instructions"]
        v = VerifierVerdict.model_validate(valid_verifier_json)
        assert v.rewrite_instructions == ""

    def test_extra_field_forbidden(self, valid_verifier_json):
        valid_verifier_json["hallucinated_flag"] = True
        with pytest.raises(ValidationError):
            VerifierVerdict.model_validate(valid_verifier_json)
