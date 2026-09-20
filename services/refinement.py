"""Turn recruiter feedback into updated criteria with traceable changes.

We send only what the model needs to make a traceable change: the current
criteria, a compact view of the shown candidates (so "1 is too junior" resolves
to a real person), and the feedback — not the whole conversation or dataset.
"""
from __future__ import annotations

import json
from typing import List

from models.schemas import RankedCandidate, RefinementResponse, SearchCriteria
from services.llm import load_prompt, structured_call


def refine(
    criteria: SearchCriteria,
    current_candidates: List[RankedCandidate],
    feedback: str,
) -> RefinementResponse:
    shown = [
        {
            "position": i + 1,  # lets feedback say "1", "2 and 4", etc.
            "id": c.id,
            "name": c.name,
            "title": c.current_title,
            "years": c.years_experience,
            "company_type": c.current_company_type,
            "score": c.score,
        }
        for i, c in enumerate(current_candidates)
    ]
    payload = {
        "current_filters": criteria.filters.model_dump(),
        "current_rubric": criteria.rubric.model_dump(),
        "currently_shown": shown,
        "recruiter_feedback": feedback,
    }
    return structured_call(
        load_prompt("refine_search.txt"),
        json.dumps(payload, ensure_ascii=False),
        RefinementResponse,
    )
