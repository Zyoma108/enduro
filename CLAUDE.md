# Enduro

LLM-driven intraday crypto trading agent. Signals come from multiple exchanges (Binance is
the source of truth by volume), execution happens on Bybit. Read `ARCHITECTURE.md` before
making structural changes — it records the agreed principles and open questions.

## Commands

- Setup: `python3 -m venv .venv && .venv/bin/pip install -e . --group dev`
- Tests: `.venv/bin/pytest`
- Lint/format: `.venv/bin/ruff check . && .venv/bin/ruff format .`
- Live data smoke test: `.venv/bin/enduro collect --symbols BTC/USDT:USDT --interval 2`

## Layout

`src/enduro/` — `core` (models), `data` (collection), `analytics`, then planned
`agent`, `risk`, `execution`, `journal`, `modes`.

## Rules

- Upper layers depend on protocols (`MarketDataSource`, future `ExecutionGateway`), never on
  ccxt directly. ccxt usage stays inside `data/ccxt_source.py` (and the future Bybit gateway).
- The risk layer is deterministic code with veto power; never move risk limits into prompts.
- Every agent decision must be journaled with its context and reasoning.
- Hot-path market models are slotted dataclasses; pydantic is for config and boundaries.
- Timestamps are epoch milliseconds; keep both `ts` (exchange) and `recv_ts` (local).
- Secrets only via env / `.env` (`ENDURO_*`), never in `config.toml` or code.
- Docs (README, ARCHITECTURE) are in Russian; code, docstrings and comments in English.
- Never place real orders in tests or during development without explicit user approval.
