"""Streamlit UI for the Sourcing Refinement Loop.

Streamlit reruns this script top-to-bottom on every interaction, so the live
SearchSession is kept in st.session_state and all logic lives in services/.
This file is presentation + wiring only.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from models.schemas import CompanyType, ObjectiveFilters, Rubric, SearchCriteria
from services import session as session_svc
from services.llm import LLMError

# st.set_page_config must be the first Streamlit command.
st.set_page_config(page_title="Sourcing Refinement Loop", page_icon="🔍", layout="wide")


def _bridge_secrets() -> None:
    """On Streamlit Cloud, copy secrets into the env the services read. Guarded on
    a secrets file existing so it stays silent (and touches nothing) locally."""
    files = [Path.home() / ".streamlit" / "secrets.toml", Path(".streamlit") / "secrets.toml"]
    if not any(p.exists() for p in files):
        return
    try:
        for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "TOP_N_CANDIDATES"):
            if k in st.secrets:
                os.environ.setdefault(k, str(st.secrets[k]))
    except Exception:
        pass


_bridge_secrets()

EXAMPLE_QUERY = (
    "Senior backend engineers in Bangalore with 5+ years, strong PostgreSQL and "
    "AWS RDS, ideally with startup experience."
)

st.markdown(
    """
    <style>
      .block-container { max-width: 1100px; }
      .app-title { font-size: 1.7rem; font-weight: 700; margin-bottom: 0; }
      .app-sub { color: #6b7280; margin-top: .15rem; }
      .sec { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
             color: #6b7280; font-weight: 700; margin: .2rem 0 .4rem; }
      .cand-name { font-size: 1.05rem; font-weight: 700; }
      .cand-meta { color: #6b7280; font-size: .85rem; }
      .pill { display:inline-block; background:#eef2ff; color:#3730a3;
              border-radius:999px; padding:2px 10px; font-size:.72rem; margin:2px 4px 2px 0; }
      .score { font-size: 1.6rem; font-weight: 800; line-height: 1; }
      .score-cap { color:#6b7280; font-size:.7rem; text-transform:uppercase; letter-spacing:.05em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- session state defaults --------------------------------------------------
st.session_state.setdefault("session", None)
st.session_state.setdefault("verdicts", {})   # candidate id -> "yes" | "no"
st.session_state.setdefault("error", None)
st.session_state.setdefault("retry", None)     # (action, arg) to re-run on failure
st.session_state.setdefault("rev", 0)          # bumps to re-seed editable widgets


# --- actions (mutate state, then rerun on success) ---------------------------
def _fail(action, arg, message):
    st.session_state.error = message
    st.session_state.retry = (action, arg)
    # Re-run so the error banner (rendered near the top) shows immediately,
    # instead of only after the recruiter's next interaction.
    st.rerun()


def do_search(query: str):
    query = query.strip()
    if not query:
        return
    try:
        with st.spinner("Interpreting your query and ranking candidates…"):
            st.session_state.session = session_svc.start_search(query)
        st.session_state.verdicts = {}
        st.session_state.error = None
        st.session_state.retry = None
        st.session_state.rev += 1
        st.rerun()
    except LLMError as e:
        _fail("search", query, e.message)


def do_apply(criteria: SearchCriteria):
    try:
        with st.spinner("Applying your changes and re-ranking…"):
            session_svc.apply_criteria(st.session_state.session, criteria)
        st.session_state.verdicts = {}
        st.session_state.error = None
        st.session_state.retry = None
        st.session_state.rev += 1
        st.rerun()
    except LLMError as e:
        _fail("apply", criteria, e.message)


def do_refine(feedback: str):
    feedback = feedback.strip()
    if not feedback:
        return
    try:
        with st.spinner("Applying your feedback and re-ranking…"):
            session_svc.refine_search(st.session_state.session, feedback)
        st.session_state.verdicts = {}
        st.session_state.pop("nl_feedback", None)
        st.session_state.error = None
        st.session_state.retry = None
        st.session_state.rev += 1
        st.rerun()
    except LLMError as e:
        _fail("refine", feedback, e.message)


def retry_last():
    action, arg = st.session_state.retry
    st.session_state.error = None
    {"search": do_search, "apply": do_apply, "refine": do_refine}[action](arg)


# --- header ------------------------------------------------------------------
st.markdown('<div class="app-title">Sourcing Refinement Loop</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Describe the talent you need. Review how the search was '
    "interpreted, then refine the shortlist with recruiter feedback.</div>",
    unsafe_allow_html=True,
)
st.write("")

# --- error banner (with retry) ----------------------------------------------
if st.session_state.error:
    c1, c2 = st.columns([5, 1])
    c1.error(st.session_state.error)
    if st.session_state.retry and c2.button("Retry", use_container_width=True):
        retry_last()


# --- helpers -----------------------------------------------------------------
def _csv(values):
    return ", ".join(values)


def _split_csv(text):
    return [v.strip() for v in text.split(",") if v.strip()]


def _lines(text):
    return [v.strip() for v in text.splitlines() if v.strip()]


def compose_feedback(session, verdicts, nl_text):
    lines = []
    for i, c in enumerate(session.candidates, 1):
        v = verdicts.get(c.id)
        if v:
            lines.append(
                f"Candidate {i} ({c.name}) is {'a good fit' if v == 'yes' else 'not a fit'}."
            )
    if nl_text.strip():
        lines.append(nl_text.strip())
    return " ".join(lines)


# ============================================================================
# INITIAL STATE — no search yet
# ============================================================================
if st.session_state.session is None:
    # max_chars bounds input client-side, so a huge paste can't run up token cost.
    query = st.text_area("What are you looking for?", key="query_input",
                         placeholder=EXAMPLE_QUERY, height=110, max_chars=2000)
    c1, c2 = st.columns([1, 5])
    if c1.button("Search", type="primary", use_container_width=True):
        do_search(st.session_state.query_input)
    # Set the value in a callback (runs before the widget re-instantiates).
    c2.button("Use example", on_click=lambda: st.session_state.update(query_input=EXAMPLE_QUERY))
    st.info(
        "Start by describing the role in plain language — location, seniority, "
        "skills, and the kind of background you want. The search will split that "
        "into objective filters (applied locally to the candidate pool) and a "
        "subjective fit rubric (used to rank), both of which you can edit."
    )
    st.stop()


# ============================================================================
# ACTIVE SEARCH
# ============================================================================
session = st.session_state.session
rev = st.session_state.rev
frozen = session.frozen

# --- sidebar: search interpretation (editable) + freeze ----------------------
with st.sidebar:
    st.markdown('<div class="sec">Original query</div>', unsafe_allow_html=True)
    st.caption(session.query)
    st.divider()

    if frozen:
        st.success("Search frozen — final criteria below.")

    f = session.criteria.filters
    r = session.criteria.rubric

    if not frozen:
        with st.form("criteria_form"):
            st.markdown('<div class="sec">Objective filters</div>', unsafe_allow_html=True)
            loc = st.text_input("Locations (comma-separated)", value=_csv(f.locations), key=f"loc_{rev}")
            cmin, cmax = st.columns(2)
            min_exp = cmin.number_input("Min years", min_value=0, max_value=50,
                                        value=f.min_experience or 0, key=f"min_{rev}")
            max_exp = cmax.number_input("Max years", min_value=0, max_value=50,
                                        value=f.max_experience or 0, key=f"max_{rev}")
            skills = st.text_input("Required skills (comma-separated)",
                                  value=_csv(f.required_skills), key=f"sk_{rev}")
            titles = st.text_input("Titles (comma-separated)", value=_csv(f.titles), key=f"ti_{rev}")
            opts = [c.value for c in CompanyType]
            cur_types = st.multiselect("Current company stage", opts,
                                       default=[c.value for c in f.current_company_types], key=f"cur_{rev}")
            any_types = st.multiselect("Stage anywhere (current or past)", opts,
                                       default=[c.value for c in f.any_company_types], key=f"any_{rev}")

            st.markdown('<div class="sec">Subjective rubric</div>', unsafe_allow_html=True)
            summary = st.text_input("Ideal candidate", value=r.summary, key=f"sum_{rev}")
            must = st.text_area("Must haves (one per line)", value="\n".join(r.must_haves), key=f"mh_{rev}", height=90)
            nice = st.text_area("Nice to haves (one per line)", value="\n".join(r.nice_to_haves), key=f"nh_{rev}", height=90)
            pen = st.text_area("Penalties (one per line)", value="\n".join(r.penalties), key=f"pn_{rev}", height=70)

            if st.form_submit_button("Apply changes & re-rank", type="primary", use_container_width=True):
                new_criteria = SearchCriteria(
                    filters=ObjectiveFilters(
                        locations=_split_csv(loc),
                        min_experience=int(min_exp) or None,
                        max_experience=int(max_exp) or None,
                        required_skills=_split_csv(skills),
                        titles=_split_csv(titles),
                        current_company_types=[CompanyType(t) for t in cur_types],
                        any_company_types=[CompanyType(t) for t in any_types],
                    ),
                    rubric=Rubric(summary=summary, must_haves=_lines(must),
                                  nice_to_haves=_lines(nice), penalties=_lines(pen)),
                )
                do_apply(new_criteria)
    else:
        # Read-only final criteria.
        st.markdown('<div class="sec">Objective filters</div>', unsafe_allow_html=True)
        st.write({
            "locations": f.locations or "—",
            "min_experience": f.min_experience,
            "max_experience": f.max_experience,
            "required_skills": f.required_skills or "—",
            "titles": f.titles or "—",
            "current_company_types": [c.value for c in f.current_company_types] or "—",
            "any_company_types": [c.value for c in f.any_company_types] or "—",
        })
        st.markdown('<div class="sec">Subjective rubric</div>', unsafe_allow_html=True)
        if r.summary:
            st.write(r.summary)
        for label, items in (("Must haves", r.must_haves), ("Nice to haves", r.nice_to_haves), ("Penalties", r.penalties)):
            if items:
                st.caption(label)
                st.write("\n".join(f"- {x}" for x in items))

    st.divider()
    if not frozen and st.button("Freeze search", use_container_width=True):
        session_svc.freeze(session)
        st.rerun()
    st.button("Start a new search", use_container_width=True,
              on_click=lambda: st.session_state.update(session=None, verdicts={}, error=None, retry=None))


# --- main: what changed (after a refine) ------------------------------------
if session.round > 0 and not frozen:
    last = session.history[-1]
    with st.container(border=True):
        st.markdown('<div class="sec">What changed</div>', unsafe_allow_html=True)
        if last.summary:
            st.write(last.summary)
        for ch in last.changes:
            st.markdown(f"**{ch.field}**: {ch.previous or '—'} → **{ch.updated or '—'}**  \n"
                        f"<span class='cand-meta'>{ch.reason}</span>", unsafe_allow_html=True)
        if not last.changes:
            st.caption("No filter/rubric changes were warranted by that feedback.")

# --- main: results header ----------------------------------------------------
head = "Final shortlist" if frozen else "Shortlist"
badge = "  ·  FINAL" if frozen else (f"  ·  round {session.round}" if session.round else "")
st.subheader(head + badge)
st.caption(f"{session.total_matched} candidate(s) matched the filters · showing top {len(session.candidates)}")

# --- main: empty states ------------------------------------------------------
if not session.candidates:
    if session.total_matched == 0:
        st.warning(
            "No candidates match these filters. Loosen a filter in the sidebar "
            "(e.g. lower the minimum years or drop a required skill), then Apply — "
            "or refine with feedback below."
        )
    else:
        st.warning("Ranking returned no candidates. Please retry, or adjust the criteria.")

# --- main: candidate cards ---------------------------------------------------
for i, c in enumerate(session.candidates, 1):
    with st.container(border=True):
        left, right = st.columns([5, 1])
        with left:
            st.markdown(f'<span class="cand-name">{i}. {c.name}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="cand-meta">{c.current_title} · {c.years_experience} yrs · {c.location}<br>'
                f'{c.current_company} ({c.current_company_type})</div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(f'<div class="score-cap">score</div><div class="score">{c.score}</div>',
                        unsafe_allow_html=True)

        st.write(c.explanation)
        if c.evidence:
            st.markdown("\n".join(f"- {e}" for e in c.evidence))
        st.markdown("".join(f'<span class="pill">{s}</span>' for s in c.skills), unsafe_allow_html=True)

        if not frozen:
            v = st.session_state.verdicts.get(c.id)
            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button(("✓ " if v == "yes" else "") + "Good fit", key=f"yes_{c.id}_{rev}",
                         type="primary" if v == "yes" else "secondary", use_container_width=True):
                st.session_state.verdicts[c.id] = "yes"
                st.rerun()
            if b2.button(("✕ " if v == "no" else "") + "Not a fit", key=f"no_{c.id}_{rev}",
                         type="primary" if v == "no" else "secondary", use_container_width=True):
                st.session_state.verdicts[c.id] = "no"
                st.rerun()

# --- main: refinement --------------------------------------------------------
if not frozen and session.candidates:
    st.write("")
    with st.container(border=True):
        st.markdown('<div class="sec">Refine with feedback</div>', unsafe_allow_html=True)
        marked = [
            f"#{i} {c.name}: {'👍' if st.session_state.verdicts[c.id] == 'yes' else '👎'}"
            for i, c in enumerate(session.candidates, 1)
            if c.id in st.session_state.verdicts
        ]
        if marked:
            st.caption("Marked: " + "   ".join(marked))
        nl = st.text_area(
            "What should change?", key="nl_feedback",
            placeholder='e.g. "1 is too junior, 2 and 4 are right" or "prefer startup experience"',
            height=90, max_chars=1000,
        )
        if st.button("Refine search", type="primary", disabled=not (marked or nl.strip())):
            do_refine(compose_feedback(session, st.session_state.verdicts, nl))

# --- main: history -----------------------------------------------------------
if session.history:
    with st.expander(f"Refinement history ({len(session.history)})"):
        for n, rec in enumerate(session.history, 1):
            st.markdown(f"**Round {n}** — _{rec.feedback}_")
            if rec.summary:
                st.caption(rec.summary)
            for ch in rec.changes:
                st.markdown(f"- {ch.field}: {ch.previous or '—'} → **{ch.updated or '—'}**")
