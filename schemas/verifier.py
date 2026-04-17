from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class VerifierVerdict(BaseModel):
    """
    Structured output of the verifier stage in the quality pipeline.

    Mirrors VERIFIER_JSON_SCHEMA in llm_client.py. rewrite_required semantics
    are defined by the verifier prompt: true when groundedness_score < 0.75
    OR unsupported_claims is non-empty OR safety_flags is non-empty.
    """

    model_config = ConfigDict(extra="forbid")

    groundedness_score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: List[str] = Field(default_factory=list)
    safety_flags: List[str] = Field(default_factory=list)
    rewrite_required: bool
    rewrite_instructions: str = ""
