from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GroundingSource(BaseModel):
    id: Optional[str] = None
    score: float = 0.0
    ts: Optional[str] = None
    preview: str


class AnalysisOutput(BaseModel):
    summary: str
    emotions: List[str] = []
    patterns: List[str] = []
    triggers: List[str] = []
    coping_suggestions: List[str] = []
    quotes_from_user: List[str] = []
    confidence: float = Field(ge=0.0, le=1.0)

    grounding_evidence: List[str] = []
    grounding_sources: List[GroundingSource] = []
    grounding_mode: Optional[str] = None
    retrieval_top_k: Optional[int] = None

    uncertainties: List[str] = []

