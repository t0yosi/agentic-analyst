# Setup and testing

A walkthrough from an empty machine to a working system and a green test suite.

## Prerequisites

| Requirement | Verify with | Notes |
|-------------|-------------|-------|
| Python 3.10+ | `python --version` | 3.12 is what this was developed against |
| pip | `pip --version` | Ships with Python |
| An Anthropic API key | `echo $ANTHROPIC_API_KEY` | Needed only to run the system, not to test it |

There is no database server to install. The analytics store is a single SQLite
file this project generates.

## One-time setup

```bash
git clone https://github.com/t0yosi/agentic-analyst.git
cd agentic-analyst

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set `ANTHROPIC_API_KEY`. Leave the other values alone unless you
have a reason to change them.

## Build the database

```bash
python main.py seed
```

Expected output:

```
Seeded bikeshare.db
  riders         600 rows
  stations        20 rows
  trips       28,391 rows
  weather        365 rows
```

The generator is seeded, so those counts are exact. Different counts mean
something changed in `database/seed.py`.

## Run the tests

The suite needs no API key and makes no network calls. Run it before you run the
system, so a failure is unambiguous.

```bash
pytest
```

Add `-v` for individual test names, or target one layer:

```bash
pytest tests/test_tools/test_sql_tools.py -v
```

## Start the server

```bash
python main.py serve
```

Expected:

```
Chat interface: http://127.0.0.1:8000
```

Open that URL. No separate web server is required; the page is served by the app.

## Check the service is healthy

```bash
curl -s localhost:8000/health
```

`database_ready` must be `true`. If it is `false`, the seed step has not run.

## Test cases

Run these in order through the chat interface.

### 1. Simple aggregate

**Ask:** *How many trips were taken in total?*

**Expect:** A single figure near 28,000. The footer shows
`call_database_agent`.

**Verify:** `[ROUTE] orchestrator -> database_agent` appears in the server log,
followed by `[ROUTE] database_agent -> execute_sql`.

### 2. Grouped comparison

**Ask:** *Which borough has the longest average trip duration?*

**Expect:** A borough name with an average in minutes, usually with the other
boroughs listed for comparison. This requires a join between `trips` and
`stations`, so it exercises schema discovery.

### 3. Chart generation

**Ask:** *Chart total trips by month.*

**Expect:** A written answer plus an embedded chart. Footer shows both
`call_database_agent` and `call_visualization_agent`, in that order, because the
chart agent can only plot data the orchestrator hands it.

### 4. Forecast with error reporting

**Ask:** *Forecast daily trips for the next 14 days.*

**Expect:** A 14-day projection **and** the holdout MAE and R squared. If the
error metrics are missing, the forecast agent's prompt is not being honoured;
that is a bug, not a stylistic issue.

**Verify:** `[LATENCY] forecast fitted n=... mae=... r2=...` in the log.

### 5. Multi-agent correlation

**Ask:** *Show me how trips trended through the year and what to expect over the
next two weeks.*

**Expect:** All three agents called. Historical figures, a projection with its
error, and a chart.

### 6. Conversation memory

**Ask:** *Which borough is busiest?* then follow with *And what about the
quietest?*

**Expect:** The second answer resolves "quietest" against the first question
without you restating the subject.

### 7. Safety guard

This one is a code-level check rather than a chat prompt, since the model will
not normally attempt it:

```bash
python -c "
from config.settings import DatabaseConfig
from tools.sql_tools import SQLToolset
SQLToolset(DatabaseConfig()).execute_sql('DROP TABLE trips')
"
```

**Expect:** `ValidationError: only SELECT or WITH queries are permitted`. The
table is untouched.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `analytics database not found` | Seed step not run | `python main.py seed` |
| `ANTHROPIC_API_KEY is not set` | No key in the environment | Set it in `.env`, or export it |
| `database_ready: false` on `/health` | Empty or missing database | Re-run the seed step |
| `LoopLimitExceeded` | Question too open-ended for the iteration budget | Ask something narrower, or raise `AgentConfig.max_iterations` |
| Chart missing from a chart request | Orchestrator answered without delegating | Ask explicitly: "chart X by Y" |
| Forecast returned with no error metrics | Prompt not honoured by the model | Check `prompts/forecast_agent/v1.txt` is being loaded; `[CONFIG] prompt loaded` in the log |
| `ModuleNotFoundError` under pytest | Virtualenv not active | `source .venv/bin/activate` |
| Tests hang or are very slow on first run | matplotlib building its font cache | Expected once; subsequent runs are fast |
