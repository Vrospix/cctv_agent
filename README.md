# Natural-Language CCTV Query Agent

A deliberately small, reliable query agent.
It translates supported natural-language requests into validated query constraints and
executes parameterized, read-only SQLite queries.

## Design

The core pipeline is intentionally deterministic:

1. Block mutation attempts and out-of-scope requests.
2. Resolve camera acronyms, full names, aliases, and conservative typo matches.
3. Convert each message into a typed `QueryPlan` before changing any state.
4. Normalize harmless language variations, then parse dates, ranges, time windows,
   recurring weekdays, sorting, and sampling into that plan.
5. Apply follow-up changes while retaining filters that were not mentioned.
6. Validate the plan and build parameterized `SELECT` and `COUNT` queries.
7. Keep pagination, clarification, and reversible search history in the chat session.

This avoids giving an LLM direct access to SQL. The optional local-AI fallback only
normalizes unusual wording into a small structured intent. The deterministic planner
still validates and translates that meaning into query constraints.

### Module responsibilities

| Module | Responsibility |
| --- | --- |
| `agent.py` | Conversation state, validation, guardrails, and responses |
| `planner.py` | Coordinates normalized text into a typed `QueryPlan` |
| `language.py` | Safe shorthand and wording normalization |
| `dates.py` / `times.py` | Date-window and clock-window extraction |
| `cameras.py` | Camera aliases, typo matching, and suggestions |
| `llm_intent.py` | Optional local JSON intent fallback through Ollama |
| `database.py` | Incremental SQLite data and parameterized read-only queries |
| `models.py` | Immutable query plans, constraints, frames, and session state |

The planner uses small pure functions for control intents, camera changes, ambiguity,
sorting, sampling, and result limits. `build_query_plan()` only coordinates those steps;
it does not execute SQL or mutate conversation state.

## Run

Python 3.11 or newer is recommended. The command-line interface has no third-party
dependencies.

```powershell
python main.py ask "Show me PIE frames between 8 AM and 10 AM yesterday."
python main.py chat
```

For the browser chat interface:

```powershell
pip install -r requirements.txt
streamlit run web_app.py
```

The browser UI preserves the same conversational agent state and displays matching
frame records as a table beneath each response. The command-line interface remains
available and has no third-party dependencies.

For a step-by-step setup and reviewer test checklist, open `USER_GUIDE.html` in a
browser.

### Optional local AI intent fallback

The application works without an AI model. For unusual wording, it can optionally use
Ollama with the approximately 986 MB `qwen2.5:1.5b` model:

```powershell
ollama pull qwen2.5:1.5b
python main.py ask "your unusual request" --local-ai
python main.py chat --local-ai
```

The Streamlit sidebar provides the same option. No API key is required, and the model
is not included in the submission. `CCTV_OLLAMA_MODEL` and `CCTV_OLLAMA_URL` can override
the model and local endpoint.

The model is called only when the deterministic planner returns `unknown`, requests
clarification, or cannot parse the wording. Ollama constrains its response to a small
JSON object with two fields: a coarse `intent` label and a normalized `meaning`. Keeping
the model's task this small is important because the default model is only 1.5B
parameters. The application then:

1. Rejects invalid JSON, extra fields, mismatched labels, and invented cameras, numbers,
   negation, dates, times, sorting, or sampling.
2. Runs the normalized meaning through the same deterministic planner.
3. Applies the existing query validation and read-only database layer.
4. Falls back to the original deterministic result if Ollama is unavailable or invalid.

The model never receives database access and never generates or executes SQL.
The smaller Qwen2.5 0.5B and Qwen3 0.6B models were also evaluated, but both produced
inconsistent labels or copied prompt wording into fields. The 1.5B model is therefore
the smallest tested option that was reliable enough for this fallback.

Inside chat, follow-ups can refine one part of the current search:

```text
Show ECP frames today
Only between 8 AM and 10 AM
Actually, use CTE instead
Also include PIE
Show the latest 10
Show one frame per hour
How many are there?
/limit 10
next
previous
What filters are active?
Undo that change
Go back to my previous search
Remove the time restriction
Start over
```

`previous` means the previous result page. `previous day` changes the active date,
while `previous search` restores the earlier set of filters.

Common chat shorthand such as `tmrw`, `yday`, `ytd`, weekday abbreviations, `@ 8am`,
and number words is normalized before planning. Negation and corrections such as
`not CTE`, `anything but PIE`, and `no CTE, use ECP` are also supported. Approximate
or contradictory wording such as `around 8ish` or `today or yesterday` triggers a
clarifying question instead of silently running a broader query.

On `ask`, `chat`, or a browser chat message, the database is automatically extended
through the current day. It adds only missing five-minute records, so normal later runs
are fast. The explicit `seed --until YYYY-MM-DD` command remains available for preparing
a particular simulated date. The generated database should be excluded from the
submission package; on the test machine, a clean first run generated approximately
80 MB of records in about two seconds.

## Test

```powershell
python -m unittest discover -s tests -v
```

## Submission package

The project can be submitted as a ZIP containing `cctv_agent/`, `tests/`, `.streamlit/`,
`main.py`, `web_app.py`, `README.md`, `USER_GUIDE.html`, and `requirements.txt`.

Generated or machine-specific files should not be included: `data/cctv_frames.db`,
`__pycache__/`, `.venv/`, local Ollama model files, and temporary files. The application
creates the database directory and simulated records automatically on its first run.

## Current limitations

- Date language is deliberately bounded rather than pretending to understand everything.
- Ambiguous or incomplete requests ask for clarification instead of guessing.
- Conversation memory and pagination are in-process only.
- Random results intentionally do not support stable next/previous pages.
- The CLI and Streamlit UI run the agent in-process; no separate HTTP API is provided.
- Local-AI interpretation is optional and safely falls back to deterministic behavior.
