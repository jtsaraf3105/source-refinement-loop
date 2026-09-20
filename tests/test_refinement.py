"""Refinement state transitions and frozen-search behavior.

The LLM parse/score/refine calls are monkeypatched — no OpenAI calls.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.schemas import (  # noqa: E402
    CandidateScore,
    FilterChange,
    ObjectiveFilters,
    RankingResponse,
    RefinementResponse,
    Rubric,
    SearchCriteria,
)
from services import ranking, refinement, session as session_svc  # noqa: E402


def _score_all(rubric, candidates):
    return RankingResponse(
        rankings=[
            CandidateScore(id=c["id"], score=80, explanation="x", evidence=["y"])
            for c in candidates
        ]
    )


def _bangalore_session(monkeypatch):
    monkeypatch.setattr(ranking, "score_candidates", _score_all)
    criteria = SearchCriteria(filters=ObjectiveFilters(locations=["Bangalore"]), rubric=Rubric())
    s = session_svc.SearchSession(query="q", criteria=criteria)
    s.candidates, s.total_matched = ranking.build_shortlist(criteria)
    return s


def test_refine_updates_criteria_and_records_history(monkeypatch):
    s = _bangalore_session(monkeypatch)

    def fake_refine(criteria, current, feedback):
        return RefinementResponse(
            filters=ObjectiveFilters(locations=["Bangalore"], min_experience=6),
            rubric=criteria.rubric,
            changes=[FilterChange(field="min_experience", previous="none", updated="6",
                                  reason="Recruiter said the shortlist was too junior.")],
            summary="Raised minimum experience to 6.",
        )

    monkeypatch.setattr(refinement, "refine", fake_refine)
    record = session_svc.refine_search(s, "too junior")

    assert s.criteria.filters.min_experience == 6
    assert s.round == 1
    assert record.changes[0].field == "min_experience"
    assert all(c.years_experience >= 6 for c in s.candidates)  # re-filtered


def test_apply_criteria_reruns_without_refine(monkeypatch):
    s = _bangalore_session(monkeypatch)
    new = SearchCriteria(filters=ObjectiveFilters(locations=["Bangalore"], min_experience=7),
                         rubric=Rubric())
    session_svc.apply_criteria(s, new)
    assert s.round == 0  # no refinement recorded
    assert all(c.years_experience >= 7 for c in s.candidates)


def test_frozen_search_cannot_be_modified(monkeypatch):
    s = _bangalore_session(monkeypatch)
    session_svc.freeze(s)
    assert s.frozen
    with pytest.raises(session_svc.FrozenSearchError):
        session_svc.refine_search(s, "change something")
    with pytest.raises(session_svc.FrozenSearchError):
        session_svc.apply_criteria(s, s.criteria)
