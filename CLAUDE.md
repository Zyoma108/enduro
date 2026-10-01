# Enduro

LLM-driven intraday crypto trading agent. Signals come from multiple exchanges (Binance is
the source of truth by volume), execution happens on Bybit. Read `ARCHITECTURE.md` before
making structural changes — it records the agreed principles and open questions.

## Commands

- Setup: `python3 -m venv .venv && .venv/bin/pip install -e . --group dev`
- Tests: `.venv/bin/pytest`
- Lint/format: `.venv/bin/ruff check . && .venv/bin/ruff format .`
- Live data smoke test: `.venv/bin/enduro collect --symbols BTC/USDT:USDT --interval 2 --no-record`
- Query recorded data: `.venv/bin/enduro sql "select count(*) from trades"`
- Radar / focus: `.venv/bin/enduro universe`, `backfill`, `scan --once`, `focus SYMBOL --json`
- Execution (demo): `.venv/bin/enduro account` (read-only); `enduro test-trade` places real
  demo orders — ask the user before running it.
- Agent: `.venv/bin/enduro agent --dry-run --ticks 2` (no orders, but each tick runs
  `claude -p` on the user's subscription — keep test runs short); without `--dry-run` it trades.

## Layout

`src/enduro/` — `core` (models), `data` (collection), `storage` (Parquet + DuckDB),
`analytics` (metrics, baselines, radar, focus), `execution` (gateway protocol + Bybit),
`risk` (limits, sizing, kill switch), `trading` (intent → exchange), `journal`, `agent`
(runtime, tools, prompt rendering, backends: Claude Code CLI via local MCP, or API).
The trader's prompt is `prompts/trader.md` — the user owns its wording.

## Rules

- Upper layers depend on protocols (`MarketDataSource`, future `ExecutionGateway`), never on
  ccxt directly. ccxt usage stays inside `data/ccxt_source.py` and `execution/bybit.py`.
- The risk layer is deterministic code with veto power; never move risk limits into prompts.
  Risk limit values are the user's decision (`[risk]` in config.toml) — don't change them
  without asking. Runtime state lives in `state/` (gitignored).
- Every agent decision must be journaled with its context and reasoning.
- Hot-path market models are slotted dataclasses; pydantic is for config and boundaries.
- Timestamps are epoch milliseconds; keep both `ts` (exchange) and `recv_ts` (local).
  Exchange clocks are skewed by tens of ms: compare exchanges only by `recv_ts`.
- Secrets only via env / `.env` (`ENDURO_*`), never in `config.toml` or code.
  Never read or print `.env`; credentials are `SecretStr` in settings.
- Order placement is never auto-retried (a timeout may still have placed it); account
  money in USDT, not Bybit's USD-denominated `totalEquity`.
- Docs (README, ARCHITECTURE) are in Russian; code, docstrings and comments in English.
- Never place real orders in tests or during development without explicit user approval.
- The user may run `enduro collect` continuously into `data/`: use `--no-record` for your own
  smoke tests so you don't write duplicate data into their store.
