# Agentic Analyst

A multi-agent analytics system that answers plain-English questions about a city
bikeshare network. An orchestrator delegates to three specialist agents, joins
what they return, and replies with written analysis and a chart.

Ask it *"how did trips trend through the year, and what should we expect over the
next two weeks?"* and it queries the database, fits a forecast, renders the chart,
and writes the answer.

## What this does

- **Natural-language to SQL** over a relational analytics database, read-only
- **Demand forecasting** with scikit-learn, reported with held-out error
- **Chart generation** returned inline as base64 PNG
- **Multi-turn conversation** with a bounded memory window
- **FastAPI service** with a chat interface, plus a CLI

All data is synthetic and generated from a fixed seed. No real riders, stations,
or trips are represented.

## Quick start

```bash
git clone https://github.com/t0yosi/agentic-analyst.git
cd agentic-analyst

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then set ANTHROPIC_API_KEY

python main.py seed         # build the synthetic database
python main.py serve        # http://127.0.0.1:8000
```

Or ask a single question without starting the server:

```bash
python main.py ask "Which borough has the longest average trip?"
```

The test suite needs no API key and makes no network calls:

```bash
pytest
```

## Architecture

```
                        User question
                              |
                    +---------v---------+
                    |    Orchestrator   |   plan -> gather -> correlate -> synthesise
                    +---------+---------+
                              |  its tools ARE the other agents
        +---------------------+---------------------+
        |                     |                     |
+-------v-------+   +---------v--------+   +--------v---------+
| Database Agent|   |  Forecast Agent  |   |Visualization Ag. |
+-------+-------+   +---------+--------+   +--------+---------+
        |                     |                     |
   SQLite (RO)          scikit-learn            matplotlib
   SELECT only          Ridge + weekday         base64 PNG
```

The orchestrator's tools are `call_database_agent`, `call_forecast_agent`, and
`call_visualization_agent`. Delegation therefore goes through the same
tool-calling mechanism as everything else, so the model decides which
specialists a given question actually needs, rather than following a fixed
pipeline.

Each specialist owns its own tools and its own system prompt. Tool-selection
accuracy degrades as a tool list grows, so separating by data domain keeps each
agent's decision space small.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed design and the reasoning
behind each decision.

## Key files

| Path | Purpose |
|------|---------|
| `agents/base.py` | The tool-use loop every agent runs |
| `agents/orchestrator.py` | Delegation, correlation, memory |
| `tools/sql_tools.py` | Read-only SQL access with statement validation |
| `tools/forecast_tools.py` | Ridge regression with held-out error reporting |
| `tools/chart_tools.py` | Chart rendering to base64 PNG |
| `llm/base.py` | Provider-agnostic model interface |
| `llm/mock_client.py` | Deterministic queue-based client for tests |
| `prompts/<component>/v1.txt` | Versioned system prompts, outside the code |
| `config/settings.py` | Typed per-layer configuration |
| `database/seed.py` | Deterministic synthetic data generator |

## Design decisions

**No agent framework.** The tool-use loop is written directly against the
provider API. For a system this size the framework abstraction costs more in
debugging opacity than it saves in code, and a hand-written loop makes the exact
sequence of tool calls inspectable in both logs and tests.

**SQL access is whitelisted, not blacklisted.** A model can propose any string,
so `tools/sql_tools.py` accepts only a single statement beginning with `SELECT`
or `WITH`, rejects every mutating keyword, and applies a row ceiling on every
call. The ceiling is documented in the tool description, because the tool schema
is the model's only instruction manual.

**The loop is bounded.** An orchestrator that delegates to agents which
themselves run tool-use loops will, given an ambiguous question, keep delegating.
`max_iterations` forces a partial answer rather than an unbounded spend.

**Forecasts always carry their error.** `forecast_daily_trips` returns holdout
MAE and R squared alongside every projection, and the agent's prompt requires
reporting them. A projection presented without its error is misleading.

**Tool failures go back to the model.** A failed tool returns an error block
rather than raising, so the model can correct a bad query instead of aborting
the run.

**Tests never call an API.** `MockLLMClient` replays a queue of explicitly
prepared responses, so tests assert on orchestration logic — which agent was
called, in what order, with what arguments — independently of what any model
happens to say on a given day.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | none | Required for `serve` and `ask` |
| `ANALYST_MODEL` | `claude-haiku-4-5` | Model identifier |
| `ANALYST_DB_PATH` | `bikeshare.db` | SQLite location |
| `ANALYST_LOG_LEVEL` | `INFO` | Logging verbosity |

## Data model

| Table | Rows | Contents |
|-------|------|----------|
| `stations` | 20 | Name, borough, capacity, coordinates |
| `riders` | 600 | Membership type, join date, birth year |
| `trips` | ~28,000 | Start and end station, timestamp, duration, distance |
| `weather` | 365 | Daily temperature, precipitation, wind |

Trip volume is generated from a seasonal curve, a weekday effect, and a weather
penalty, so the series has structure a forecast can actually learn.

## License

MIT
