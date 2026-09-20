# Sourcing Refinement Loop

A recruiter describes who they want in plain language. The app turns that into
**objective filters** + a **subjective fit rubric**, filters the 48 supplied
profiles locally, asks an LLM to rank the survivors with grounded explanations,
and lets the recruiter **refine with natural-language and per-candidate feedback**
over multiple rounds before **freezing** a final shortlist.

Built for the Flexiple engineering assignment. Scope is deliberately a focused
take-home: no auth, no database, no infra — a Streamlit UI over a few Python
services, with in-memory session state.

```
NL query → LLM parses filters + rubric → recruiter edits →
local filtering → LLM ranks → 4–5 cards with explanations →
recruiter feedback → LLM refines (traceable changes) → re-run → … → freeze
```

---

## Problem being solved

A recruiter working a large talent pool needs to go from a vague requirement to a
short, defensible shortlist. Keyword search alone can't judge *fit*; asking an LLM
to do everything is expensive and hallucination-prone. This app splits the work:
deterministic Python decides *who objectively qualifies*, and the LLM decides
*who best fits* and *how the search should change* given feedback.

## Architecture

```
Streamlit UI (app.py)
      │  st.session_state holds the live SearchSession
      ▼
Python services/
      ├─ session.py     start_search / apply_criteria / refine_search / freeze
      ├─ ranking.py     parse_query, score_candidates, build_shortlist
      ├─ refinement.py  refine (feedback → new criteria + changes)
      ├─ filtering.py   deterministic local filtering
      └─ llm.py         one OpenAI structured-output wrapper + error handling
      ▼
OpenAI  (parse / rank / refine)      data/profiles.json  (deterministic filter)
```

No HTTP layer: Streamlit runs server-side and calls the services directly. The
OpenAI key stays on the server and never reaches the browser.

## Search / refinement flow

1. Recruiter enters a query → `ranking.parse_query` (LLM) → `SearchCriteria`.
2. `filtering.apply_filters` narrows the 48 profiles deterministically.
3. `ranking.score_candidates` (LLM) scores only the survivors → top N cards.
4. Recruiter edits criteria (re-run, no LLM refine) and/or gives feedback.
5. `refinement.refine` (LLM) returns updated criteria + a list of traceable
   changes; the shortlist re-runs and the UI shows *what changed and why*.
6. Repeat, then **Freeze** to lock the final filters, rubric, and shortlist.

## Why Streamlit

The app is a single-user, single-session workflow over 48 profiles. Streamlit
lets the whole thing stay Python — the UI, state, and services in one language —
which keeps the focus on the AI/search problem and makes every line explainable.
It also deploys in one step (Streamlit Community Cloud). A React SPA would add a
second language and a separate API for no benefit at this size.

## Why deterministic filtering + LLM ranking

`years >= 5` or skill-set membership are exact, free, and reproducible — Python
does them. Judging "strong PostgreSQL depth, startup mindset" is subjective — the
LLM does that, but only on the already-filtered set, and only using fields we send
it, so explanations stay grounded in real profile data.

## Why no database

The dataset is a static 48-row JSON file loaded once, and sessions are short
refinement conversations. State lives in `st.session_state`. A database would be
ceremony here; the scaling notes below say where it enters at production scale.

## LLM usage

Three prompts (in `prompts/`), each called via `services/llm.py::structured_call`
with a Pydantic schema as the response format, so the model is forced to return
valid JSON we validate:

- `parse_search.txt` — query → objective filters + rubric.
- `score_candidates.txt` — rubric + filtered profiles → scores/evidence.
- `refine_search.txt` — criteria + feedback → updated criteria + changes.

Calls per action: **initial search = 2** (parse + rank); **refinement = 2**
(refine + rank); **manual criteria edit = 1** (rank only); **empty filter result
= 0** (LLM skipped).

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your key:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### Environment variables

| Var                | Default        | Purpose |
|--------------------|----------------|---------|
| `OPENAI_API_KEY`   | *(required)*   | Server-side key. `.env` is gitignored. |
| `OPENAI_MODEL`     | `gpt-4o-mini`  | Any structured-output-capable chat model. Change this line alone to switch models. |
| `TOP_N_CANDIDATES` | `5`            | How many cards to show. |
| `OPENAI_TIMEOUT`   | `40`           | Per-request timeout (seconds). |

On Streamlit Community Cloud, put the same keys in the app's **Secrets**; `app.py`
bridges them into the environment automatically.

## Running locally

```bash
streamlit run app.py
```

Opens at http://localhost:8501.

**Example query:** *"Senior backend engineers in Bangalore with 5+ years, strong
PostgreSQL and AWS RDS, ideally with startup experience."*
**Example refinement:** *"1 is too junior, 2 and 4 are right, prefer startup
background."*

## Testing

```bash
pytest -q
```

Covers filtering (experience/location/skill/company-stage/combined/empty), the
ranking pipeline (sort + top-N + empty-skip + omitted-candidate handling), the
refinement state transition, the frozen-search guard, and malformed-LLM-output
handling. The LLM is monkeypatched — **tests never call OpenAI**.

## Failure handling

Every OpenAI failure mode (bad key, timeout, rate limit, network, malformed/empty
output, refusal) is caught in `llm.py` and re-raised as a single user-safe
`LLMError`. The UI shows a red banner with a **Retry** that re-runs the last
action; the failure surfaces immediately (the action re-runs so the banner is
shown right away, not on the next click), and the technical detail is logged
server-side, never shown to the recruiter. Empty states are handled explicitly:
initial (no search), no objective matches, no ranking result, and frozen.

Recruiter inputs are length-capped client-side (query 2000 chars, feedback 1000)
so a large paste can't run up token cost; a normal search is a fraction of a cent.

## Key trade-offs

- **In-memory session state** — simplest thing that works; lost on restart.
  Isolated in `session.py`, so a Redis/Postgres store is a localized change.
- **LLM ranking** — subjective fit needs judgement, but scores vary slightly run
  to run; mitigated with `temperature=0.2` and a fixed 0–100 scale, and used only
  to *order* one shortlist, not as a calibrated metric.
- **Parse can over-constrain** (e.g. treat a title as a hard filter) — which is
  exactly why filters are editable in the UI.

## Known limitations

- Sessions vanish on restart (by design).
- No pagination; always shows the top N of whatever matched.
- The 48 profiles are intentionally the entire pool for this exercise — the app is
  **not** built to handle a large talent pool as-is.

## Future production considerations

Not built here — this is the 48-profile exercise — but the shape already scales:
keep deterministic pre-filtering first (in indexed SQL, not Python), add a cheap
first-pass ranker (keyword/embeddings) so the LLM only scores a small top slice,
then cache scores, batch calls, move long jobs async with polling, and add
observability (tokens/latency/cost), rate limiting, and an offline ranking-quality
eval set. See DESIGN_DECISIONS.md and INTERVIEW_PREP.md.
