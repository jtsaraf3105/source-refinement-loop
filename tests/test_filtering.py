"""Deterministic filtering — the part that can silently return the wrong
candidates. Uses the real profiles.json dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.schemas import CompanyType, ObjectiveFilters  # noqa: E402
from services.filtering import apply_filters, load_profiles  # noqa: E402


def test_dataset_loads():
    assert len(load_profiles()) == 48


def test_min_experience():
    out = apply_filters(ObjectiveFilters(min_experience=10))
    assert out and all(p["years_experience"] >= 10 for p in out)


def test_experience_range():
    out = apply_filters(ObjectiveFilters(min_experience=5, max_experience=7))
    assert all(5 <= p["years_experience"] <= 7 for p in out)


def test_location_is_case_insensitive():
    out = apply_filters(ObjectiveFilters(locations=["bangalore"]))
    assert out and all(p["location"] == "Bangalore" for p in out)


def test_required_skills_are_conjunctive_and_case_insensitive():
    out = apply_filters(ObjectiveFilters(required_skills=["postgresql", "AWS RDS"]))
    assert out
    for p in out:
        have = {s.lower() for s in p["skills"]}
        assert "postgresql" in have and "aws rds" in have


def test_current_company_type():
    out = apply_filters(ObjectiveFilters(current_company_types=[CompanyType.startup]))
    assert out and all(p["current_company_type"] == "startup" for p in out)


def test_any_company_type_checks_past_experience():
    # p01 is currently at a startup but previously at Freshworks (scaleup).
    out = apply_filters(ObjectiveFilters(any_company_types=[CompanyType.scaleup]))
    assert "p01" in {p["id"] for p in out}


def test_combined_filters_narrow_results():
    broad = apply_filters(ObjectiveFilters(locations=["Bangalore"]))
    narrow = apply_filters(
        ObjectiveFilters(locations=["Bangalore"], min_experience=6, required_skills=["AWS RDS"])
    )
    assert len(narrow) <= len(broad)
    for p in narrow:
        assert p["location"] == "Bangalore" and p["years_experience"] >= 6


def test_empty_result_when_no_match():
    assert apply_filters(ObjectiveFilters(locations=["Atlantis"])) == []


def test_no_filters_returns_everyone():
    assert len(apply_filters(ObjectiveFilters())) == 48
