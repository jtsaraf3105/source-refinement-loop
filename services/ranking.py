"""LLM query-parsing and candidate ranking, plus the small pipeline that turns
criteria into a ranked shortlist.

Cost discipline: only the FILTERED candidates are sent to the LLM, trimmed to
the fields the rubric can actually use.
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

from config import get_settings
from models.schemas import RankedCandidate, RankingResponse, Rubric, SearchCriteria
from services.filtering import apply_filters
from services.llm import load_prompt, structured_call


def parse_query(query: str) -> SearchCriteria:
    """Natural language -> objective filters + subjective rubric."""
    return structured_call(load_prompt("parse_search.txt"), query, SearchCriteria)


def _compact(profile: dict) -> dict:
    """Only the fields the LLM needs to judge fit — keeps the prompt small and
    forces evidence to come from real data."""
    return {
        "id": profile["id"],
        "title": profile["current_title"],
        "years": profile["years_experience"],
        "location": profile["location"],
        "company": profile["current_company"],
        "company_type": profile["current_company_type"],
        "skills": profile["skills"],
        "past": [
            {"company": p["company"], "type": p["company_type"], "years": p["years"]}
            for p in profile.get("past_companies", [])
        ],
    }


def score_candidates(rubric: Rubric, candidates: List[dict]) -> RankingResponse:
    """Score the pre-filtered candidates against the rubric."""
    payload = {"rubric": rubric.model_dump(), "candidates": [_compact(c) for c in candidates]}
    return structured_call(
        load_prompt("score_candidates.txt"),
        json.dumps(payload, ensure_ascii=False),
        RankingResponse,
    )


def build_shortlist(
    criteria: SearchCriteria, top_n: Optional[int] = None
) -> Tuple[List[RankedCandidate], int]:
    """Deterministic filter -> LLM score -> top-N joined cards.

    Returns (cards, total_matched). Skips the LLM entirely when filtering yields
    nothing (empty state, saves a call).
    """
    if top_n is None:
        top_n = get_settings().top_n

    matched = apply_filters(criteria.filters)
    if not matched:
        return [], 0

    scores = {s.id: s for s in score_candidates(criteria.rubric, matched).rankings}

    cards: List[RankedCandidate] = []
    for profile in matched:
        s = scores.get(profile["id"])
        if s is None:
            continue  # model omitted this candidate; skip rather than fabricate
        cards.append(
            RankedCandidate(
                id=profile["id"],
                name=profile["name"],
                current_title=profile["current_title"],
                years_experience=profile["years_experience"],
                location=profile["location"],
                current_company=profile["current_company"],
                current_company_type=profile["current_company_type"],
                skills=profile["skills"],
                score=s.score,
                explanation=s.explanation,
                evidence=s.evidence,
            )
        )

    cards.sort(key=lambda c: c.score, reverse=True)
    return cards[:top_n], len(matched)
