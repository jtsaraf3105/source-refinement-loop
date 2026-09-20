"""Deterministic, local filtering of profiles.json.

Filtering is exact, cheap, and reproducible, so there is no reason to spend
tokens (or risk hallucination) on it. The LLM decides *what* to filter on; this
module *applies* it.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

from models.schemas import ObjectiveFilters

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "profiles.json"


@lru_cache
def load_profiles() -> List[dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm(value: str) -> str:
    return value.strip().lower()


def _company_types(profile: dict) -> set[str]:
    """All company stages in a profile: current + every past company."""
    types = {_norm(profile.get("current_company_type", ""))}
    for past in profile.get("past_companies", []):
        types.add(_norm(past.get("company_type", "")))
    types.discard("")
    return types


def _matches(profile: dict, f: ObjectiveFilters) -> bool:
    exp = profile.get("years_experience", 0)
    if f.min_experience is not None and exp < f.min_experience:
        return False
    if f.max_experience is not None and exp > f.max_experience:
        return False

    if f.locations:
        wanted = {_norm(l) for l in f.locations}
        if _norm(profile.get("location", "")) not in wanted:
            return False

    if f.titles:
        wanted = {_norm(t) for t in f.titles}
        if _norm(profile.get("current_title", "")) not in wanted:
            return False

    if f.required_skills:
        have = {_norm(s) for s in profile.get("skills", [])}
        for skill in f.required_skills:
            if _norm(skill) not in have:
                return False

    if f.current_company_types:
        wanted = {ct.value for ct in f.current_company_types}
        if _norm(profile.get("current_company_type", "")) not in wanted:
            return False

    if f.any_company_types:
        wanted = {ct.value for ct in f.any_company_types}
        if not (wanted & _company_types(profile)):
            return False

    return True


def apply_filters(filters: ObjectiveFilters) -> List[dict]:
    """Return the profiles that satisfy ALL active filters."""
    return [p for p in load_profiles() if _matches(p, filters)]
