"""Pydantic models — the single source of truth for every LLM structured output
and the in-app data shapes. Passing these classes to the OpenAI SDK forces the
model to return exactly these fields; anything malformed fails validation here.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class CompanyType(str, Enum):
    startup = "startup"
    scaleup = "scaleup"
    enterprise = "enterprise"
    agency = "agency"


# --- Search criteria: objective filters + subjective rubric ------------------
class ObjectiveFilters(BaseModel):
    """Hard requirements evaluated deterministically. Empty list / None on a
    field means 'don't constrain on this dimension'."""

    locations: List[str] = Field(default_factory=list)
    min_experience: Optional[int] = None
    max_experience: Optional[int] = None
    required_skills: List[str] = Field(default_factory=list)
    titles: List[str] = Field(default_factory=list)
    current_company_types: List[CompanyType] = Field(default_factory=list)
    any_company_types: List[CompanyType] = Field(default_factory=list)


class Rubric(BaseModel):
    """Subjective fit signals — used to rank, never to filter."""

    summary: str = ""
    must_haves: List[str] = Field(default_factory=list)
    nice_to_haves: List[str] = Field(default_factory=list)
    penalties: List[str] = Field(default_factory=list)


class SearchCriteria(BaseModel):
    filters: ObjectiveFilters
    rubric: Rubric


# --- Ranking -----------------------------------------------------------------
class CandidateScore(BaseModel):
    """One scored candidate. `evidence` must quote real profile fields."""

    id: str
    score: int = Field(ge=0, le=100)
    explanation: str
    evidence: List[str] = Field(default_factory=list)


class RankingResponse(BaseModel):
    rankings: List[CandidateScore]


# --- Refinement --------------------------------------------------------------
class FilterChange(BaseModel):
    """A single traceable change made by refinement, with its reason."""

    field: str
    previous: str
    updated: str
    reason: str


class RefinementResponse(BaseModel):
    filters: ObjectiveFilters
    rubric: Rubric
    changes: List[FilterChange] = Field(default_factory=list)
    summary: str = ""


# --- Candidate shown in the UI (score joined with profile fields) ------------
class RankedCandidate(BaseModel):
    id: str
    name: str
    current_title: str
    years_experience: int
    location: str
    current_company: str
    current_company_type: str
    skills: List[str]
    score: int
    explanation: str
    evidence: List[str]
