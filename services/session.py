"""The search session: state + the transitions that move it forward.

Streamlit reruns the whole script on every interaction, so the live SearchSession
object is stored in st.session_state (see app.py). This module is UI-agnostic:
it just holds the state and the operations (start / apply edits / refine / freeze),
which keeps the business logic testable without Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from models.schemas import FilterChange, RankedCandidate, SearchCriteria
from services import ranking, refinement


class FrozenSearchError(Exception):
    """Raised when something tries to modify a frozen search."""


@dataclass
class RefinementRecord:
    feedback: str
    changes: List[FilterChange]
    summary: str


@dataclass
class SearchSession:
    query: str
    criteria: SearchCriteria
    candidates: List[RankedCandidate] = field(default_factory=list)
    total_matched: int = 0
    history: List[RefinementRecord] = field(default_factory=list)
    frozen: bool = False

    @property
    def round(self) -> int:
        # Round 0 = initial search; each refinement bumps it.
        return len(self.history)


def start_search(query: str) -> SearchSession:
    """Parse the query, filter, and rank — the initial search."""
    criteria = ranking.parse_query(query)
    session = SearchSession(query=query, criteria=criteria)
    session.candidates, session.total_matched = ranking.build_shortlist(criteria)
    return session


def apply_criteria(session: SearchSession, criteria: SearchCriteria) -> None:
    """Re-run after the recruiter hand-edits the criteria — no LLM refine call."""
    if session.frozen:
        raise FrozenSearchError("This search is frozen.")
    session.criteria = criteria
    session.candidates, session.total_matched = ranking.build_shortlist(criteria)


def refine_search(session: SearchSession, feedback: str) -> RefinementRecord:
    """Refine criteria from feedback (LLM), then re-filter and re-rank."""
    if session.frozen:
        raise FrozenSearchError("This search is frozen.")
    result = refinement.refine(session.criteria, session.candidates, feedback)
    session.criteria = SearchCriteria(filters=result.filters, rubric=result.rubric)
    session.candidates, session.total_matched = ranking.build_shortlist(session.criteria)
    record = RefinementRecord(feedback=feedback, changes=result.changes, summary=result.summary)
    session.history.append(record)
    return record


def freeze(session: SearchSession) -> None:
    session.frozen = True
