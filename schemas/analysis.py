from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GroundingSource(BaseModel):
    """One retrieved source referenced in an analysis."""

    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = None
    score: float = 0.0
    ts: Optional[str] = None
    preview: str


class AnalysisOutput(BaseModel):
    """
    Structured analysis produced by the draft/revise stages.

    Required fields match DRAFT_JSON_SCHEMA in llm_client.py. The grounding_*
    and uncertainties fields are optional enrichment emitted by the LangChain
    insight chain; they default to empty and do not need to be present in
    Ollama responses to pass validation.

    extra="forbid" guards against models hallucinating additional keys.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    emotions: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    coping_suggestions: List[str] = Field(default_factory=list, max_length=10)
    quotes_from_user: List[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)

    grounding_evidence: List[str] = Field(default_factory=list)
    grounding_sources: List[GroundingSource] = Field(default_factory=list)
    grounding_mode: Optional[str] = None
    retrieval_top_k: Optional[int] = Field(default=None, ge=0)

    uncertainties: List[str] = Field(default_factory=list)
