"""Candle persistence.

Layout: <root>/candles/timeframe=<tf>/exchange=<exchange>/part-<first_ts>-<rand>.parquet
Candles are small (one row per symbol per minute), so they are not partitioned by date.
Only closed candles are ever written, so appending from the last stored `ts` never
creates duplicates.
"""

from __future__ import annotations

import secrets
from collections import defaultdict
from pathlib import Path

import pyarrow as pa

from enduro.core.models import Candle
from enduro.storage import store
from enduro.storage.files import write_parquet_atomic

CANDLE_SCHEMA = pa.schema(
    [
        ("ts", pa.int64()),
        ("exchange", pa.string()),
        ("symbol", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
    ]
)


def write_candles(root: Path | str, candles: list[Candle], timeframe: str = "1m") -> int:
    """Write candles, one file per exchange. Returns the number of rows written."""
    by_exchange: dict[str, list[Candle]] = defaultdict(list)
    for c in candles:
        by_exchange[c.exchange].append(c)
    for exchange, rows in by_exchange.items():
        directory = Path(root) / "candles" / f"timeframe={timeframe}" / f"exchange={exchange}"
        directory.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(
            [
                {
                    "ts": c.ts,
                    "exchange": c.exchange,
                    "symbol": c.symbol,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in rows
            ],
            schema=CANDLE_SCHEMA,
        ).sort_by([("symbol", "ascending"), ("ts", "ascending")])
        path = directory / f"part-{min(c.ts for c in rows)}-{secrets.token_hex(4)}.parquet"
        write_parquet_atomic(table, path)
    return len(candles)


def last_candle_ts(root: Path | str, exchange: str, timeframe: str = "1m") -> dict[str, int]:
    """Open time of the newest stored candle per symbol on `exchange`."""
    con = store.connect(root)
    if not store.has_view(con, "candles"):
        return {}
    rows = con.execute(
        "SELECT symbol, max(ts) FROM candles WHERE exchange = ? AND timeframe = ? GROUP BY 1",
        [exchange, timeframe],
    ).fetchall()
    return dict(rows)


def load_recent_candles(
    root: Path | str, exchange: str, symbols: list[str], since_ms: int, timeframe: str = "1m"
) -> list[Candle]:
    con = store.connect(root)
    if not store.has_view(con, "candles"):
        return []
    rows = con.execute(
        "SELECT exchange, symbol, ts, open, high, low, close, volume FROM candles "
        "WHERE exchange = ? AND timeframe = ? AND ts >= ? AND list_contains(?, symbol) "
        "ORDER BY symbol, ts",
        [exchange, timeframe, since_ms, symbols],
    ).fetchall()
    return [Candle(*row) for row in rows]
