"""Pydantic schemas for structured agent outputs."""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class ClaimsOutput(BaseModel):
    claims: List[str] = Field(
        ..., description="A list of 3 concise, verifiable factual claims from the article."
    )


Verdict = Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]


class ClaimVerdict(BaseModel):
    claim: str
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str = Field(..., description="Short explanation citing the Wikipedia source used.")


class FactCheckOutput(BaseModel):
    results: List[ClaimVerdict]


class BiasOutput(BaseModel):
    tone: str = Field(..., description="Single-word tone label, e.g. neutral, sensationalist, mocking.")
    bias_score: float = Field(..., ge=0.0, le=1.0, description="0 = neutral, 1 = highly biased")
    flags: List[str] = Field(default_factory=list)


class AgentInputsSummary(BaseModel):
    claim_verdicts: List[Verdict] = Field(default_factory=list)
    bias_score: float = 0.0
    tone: str = "unknown"


class JudgeOutput(BaseModel):
    label: Literal["REAL", "FAKE"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    agent_inputs_summary: AgentInputsSummary
