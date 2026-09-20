# Sourcing Refinement Loop

A recruiter describes who they want in plain language. The app converts the
request into **objective filters** and a **subjective fit rubric**, filters the
48 supplied profiles locally, uses an LLM to rank the matching candidates, and
lets the recruiter refine the search through natural-language and
per-candidate feedback before freezing the final shortlist.

Built for the Flexiple Engineering Challenge.

## Flow

```text
Recruiter query
      ↓
LLM → objective filters + subjective rubric
      ↓
Recruiter can edit filters/rubric
      ↓
Deterministic local filtering
      ↓
LLM ranking of filtered candidates
      ↓
Top 4–5 candidates with grounded explanations
      ↓
Recruiter feedback / refinement
      ↓
Updated filters + "What changed"
      ↓
Re-run → repeat → Freeze
```

## Architecture

```text
Streamlit UI
     ↓
Python services
 ├── session.py
 ├── ranking.py
 ├── filtering.py
 ├── refinement.py
 └── llm.py
     ↓
OpenAI API + profiles.json
```

The app uses Streamlit with in-memory session state. There is no database or
separate HTTP API because the assignment uses a fixed dataset of 48 profiles
and requires a focused single-session workflow.

## Key Design Decisions

### Objective filtering vs. LLM ranking

Deterministic criteria such as experience, location, skills, and company stage
are applied locally in Python. This keeps objective filtering predictable and
cheap.

The LLM is used after filtering to judge subjective fit against the rubric and
explain why each candidate matches.

### Grounded candidate explanations

Only candidates that pass the objective filters are sent to the ranking LLM,
using relevant fields from the supplied profiles. The ranking prompt requires
evidence based on those profile fields.

### Structured LLM output

LLM responses use Pydantic schemas and are validated before being used by the
application.

### Refinement loop

Recruiter feedback is sent to the LLM together with the current criteria and
shown candidates. The LLM returns updated criteria and explicit changes so the
UI can show what changed and why.

### Error handling

LLM failures such as invalid API keys, timeouts, rate limits, connection
errors, malformed responses, and refusals are handled through a common
`LLMError`. The UI provides a retry path.

## Setup

Requires Python 3.13.

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Run:

```bash
streamlit run app.py
```

The application opens at:

```text
http://localhost:8501
```

## Testing

```bash
pytest -q
```

Tests cover deterministic filtering, ranking behavior, refinement state
transitions, frozen searches, empty results, and malformed LLM output.

The tests mock the LLM, so they do not require OpenAI API calls.

## Assignment Scope

Intentionally not included:

- Authentication
- Database/persistence
- Multiple user roles
- External integrations
- Production infrastructure

These are outside the scope of the assignment.

For a larger production talent pool, the filtering layer could move to an
indexed datastore, followed by a cheaper retrieval/ranking stage before
calling the LLM.
