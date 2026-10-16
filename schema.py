"""Typed outputs every agent must return."""
from typing import Literal
from pydantic import BaseModel, Field

Verdict = Literal["confirmed", "dismissed", "inconclusive"]


class Scenario(BaseModel):
    scenario_id: str = Field(description="short id like S1")
    description: str
    duration_months: int = Field(default=6, description="how long the shock lasts; 0 for one-off events")


class ScenarioList(BaseModel):
    scenarios: list[Scenario]


class Citation(BaseModel):
    ref: str = Field(description="chunk reference exactly as given, e.g. term_policy.md#5.2")
    quote: str = Field(description="short exact phrase copied from that chunk")


class Claim(BaseModel):
    position: Literal["gap", "no_gap"]
    argument: str
    citations: list[Citation] = []


class Ruling(BaseModel):
    verdict: Verdict
    justification: str
    decisive_refs: list[str] = []


class Report(BaseModel):
    summary: str
    findings: list[str]
