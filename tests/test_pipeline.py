"""The filter→rank pipeline (build_shortlist) and LLM error handling.

The LLM is monkeypatched — tests never call OpenAI. We verify the deterministic
glue: sort + top-N truncation, the empty-skip, and malformed-output handling.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.schemas import (  # noqa: E402
    CandidateScore,
    ObjectiveFilters,
    RankingResponse,
    Rubric,
    SearchCriteria,
)
from services import ranking  # noqa: E402
from services.llm import LLMError, structured_call  # noqa: E402


def _criteria(**filters):
    return SearchCriteria(filters=ObjectiveFilters(**filters), rubric=Rubric())


def test_build_shortlist_sorts_and_truncates(monkeypatch):
    def fake_score(rubric, candidates):
        # Ascending score by list order, so we can check it gets re-sorted desc.
        return RankingResponse(
            rankings=[
                CandidateScore(id=c["id"], score=i, explanation="x", evidence=["y"])
                for i, c in enumerate(candidates)
            ]
        )

    monkeypatch.setattr(ranking, "score_candidates", fake_score)
    cards, total = ranking.build_shortlist(_criteria(locations=["Bangalore"]), top_n=5)

    assert total > 5
    assert len(cards) == 5  # truncated to top N
    scores = [c.score for c in cards]
    assert scores == sorted(scores, reverse=True)  # highest first


def test_build_shortlist_empty_skips_llm(monkeypatch):
    monkeypatch.setattr(
        ranking, "score_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )
    cards, total = ranking.build_shortlist(_criteria(locations=["Atlantis"]))
    assert cards == [] and total == 0


def test_build_shortlist_skips_candidates_the_model_omitted(monkeypatch):
    # Model scores only one of the matched candidates; the rest are dropped, not faked.
    def partial_score(rubric, candidates):
        first = candidates[0]["id"]
        return RankingResponse(
            rankings=[CandidateScore(id=first, score=90, explanation="x", evidence=["y"])]
        )

    monkeypatch.setattr(ranking, "score_candidates", partial_score)
    cards, total = ranking.build_shortlist(_criteria(locations=["Bangalore"]))
    assert len(cards) == 1 and total > 1


def test_malformed_llm_output_raises_clean_error(monkeypatch):
    class _Msg:
        refusal = None
        parsed = None  # simulate empty/unparseable structured output

    class _Completion:
        choices = [type("C", (), {"message": _Msg()})]

    class _FakeClient:
        class beta:
            class chat:
                class completions:
                    @staticmethod
                    def parse(**kwargs):
                        return _Completion()

    monkeypatch.setattr("services.llm._client", lambda: _FakeClient())
    with pytest.raises(LLMError):
        structured_call("sys", "user", RankingResponse)
