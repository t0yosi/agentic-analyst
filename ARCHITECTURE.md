# Architecture

## Request lifecycle

1. A question arrives at `POST /ask` or through `python main.py ask`.
2. `Orchestrator.ask` prepends conversation memory and enters the tool-use loop.
3. The model chooses which specialist agents to call, and in what order.
4. Each delegation runs that agent's own full tool-use loop and returns prose.
5. The orchestrator correlates the returns and writes the final answer.
6. Any chart rendered along the way is attached as base64 PNG.

## Layers

```
api/          FastAPI routes, request and response schemas, chat page
agents/       Orchestrator and three specialists, all over one shared loop
tools/        SQL, forecasting, and charting, each a self-contained toolset
llm/          Provider-agnostic client interface, Anthropic impl, mock impl
prompts/      Versioned system prompts as text files
database/     Schema, connection handling, deterministic synthetic seeder
config/       Typed dataclass configuration per layer
utils/        Exception hierarchy and logging setup
```

Dependencies point one way: `api` depends on `agents`, `agents` depends on
`tools` and `llm`, and `tools` depends on `database`. Nothing lower reaches up.

## The tool-use loop

`agents/base.py` implements the loop once. Per iteration:

1. Call the model with the conversation, system prompt, and tool definitions.
2. If no tool was requested, the turn is final. Return the text.
3. Otherwise append the assistant turn, execute each requested tool, and append
   the results as a user turn.
4. Repeat until `max_iterations`, then raise `LoopLimitExceeded`.

Two properties matter. **Tool errors do not abort the run** — they return as
`is_error` blocks so the model can correct itself, which is what turns a
malformed query into a retry rather than a failure. And **the loop is bounded**,
because a delegating orchestrator whose specialists also loop can otherwise
recurse indefinitely on an ambiguous question.

## Agents as tools

The orchestrator holds no data tools. Its three tools are the specialists, and
`_AgentDelegationToolset.dispatch` runs the target agent's full loop and returns
its final text.

Consequences worth stating:

- A specialist cannot see the user's original question or another specialist's
  reply. The orchestrator prompt therefore requires self-contained instructions.
- Adding a specialist is adding a tool definition. No routing code changes.
- The call sequence is observable, which is what the orchestrator tests assert on.

## Data access safety

`tools/sql_tools.py` enforces, in order:

1. The statement must begin with `SELECT` or `WITH`.
2. It must not contain any mutating keyword, which catches `SELECT 1; DROP TABLE`.
3. It must be a single statement.
4. Results are truncated at `DatabaseConfig.max_rows`.

Connections are additionally opened with SQLite's `mode=ro` URI, so the guard is
defence in depth rather than the only barrier. In a production deployment the
next layer is a database user that holds only `SELECT` grants; application-level
validation should never be the sole control.

Identifiers passed to `get_table_sample` are matched against `^[A-Za-z_]\w*$` and
then checked against `sqlite_master`, because that argument is interpolated
rather than bound.

## Forecasting

`forecast_daily_trips` aggregates trips per day, builds a design matrix of a
trend index plus a one-hot weekday encoding, and fits a ridge regression. The
series is split chronologically, never shuffled, since shuffling a time series
leaks the future into training. Held-out MAE and R squared are computed on the
tail, the model is refit on the full series, and the projection is clipped at
zero.

The agent's prompt requires reporting the error metrics with any projection.

## Configuration

Each layer owns a dataclass validating its own values in `__post_init__`, so an
invalid setting fails at construction with a message naming the bad value and
the constraint. `AppConfig.log_active_config` emits every active hyperparameter
under `[CONFIG]`, which makes a run's logs self-contained.

## Observability

Structured, greppable log tags:

| Tag | Emitted at |
|-----|-----------|
| `[CONFIG]` | Startup, once per layer, plus prompt version resolution |
| `[STAGE]` | Agent start and finish, schema application, chart render |
| `[ROUTE]` | Every delegation and every tool dispatch, with arguments |
| `[DATA]` | Row counts from queries, seeding, schema description |
| `[TOKENS]` | Per model call and per completed run |
| `[LATENCY]` | Forecast fitting, with sample size and error |

## Testing

`MockLLMClient` replays an explicitly queued sequence of responses. A test states
exactly what the model will do, so assertions target orchestration logic rather
than model output. The suite makes no network calls and needs no API key.

Coverage is organised by layer, mirroring the source tree under `tests/`.

## Known limits

- **Schema size.** The schema is small enough to pass whole to the model. A real
  warehouse needs retrieval over the schema itself.
- **Single database.** One SQLite file. Multiple sources would need one agent per
  source, which is exactly why tools are grouped by data domain already.
- **Forecast model.** Ridge with trend and weekday effects captures the seasonal
  shape but not holidays, events, or weather at prediction time. The weather
  table is present and joinable, so weather-conditioned forecasting is the
  obvious next step.
- **Memory is in-process.** Conversation history lives on the orchestrator
  instance and does not survive a restart or span workers.
