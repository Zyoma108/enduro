"""Persists market events from the bus into Hive-partitioned Parquet files.

Layout: <root>/<kind>/date=YYYY-MM-DD/exchange=<exchange>/part-<first_ts>-<rand>.parquet
where kind is "trades" or "books" and the date is the UTC date of the event.

Every trade is stored. Order books update tens of times per second, so book
snapshots are downsampled to at most one per `book_interval_ms` per (exchange, symbol).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from enduro.core.models import MarketEvent, OrderBook, Trade
from enduro.storage.files import write_parquet_atomic

log = logging.getLogger(__name__)

TRADE_SCHEMA = pa.schema(
    [
        ("ts", pa.int64()),
        ("recv_ts", pa.int64()),
        ("exchange", pa.string()),
        ("symbol", pa.string()),
        ("price", pa.float64()),
        ("amount", pa.float64()),
        ("side", pa.string()),
        ("id", pa.string()),
    ]
)

BOOK_SCHEMA = pa.schema(
    [
        ("ts", pa.int64()),
        ("recv_ts", pa.int64()),
        ("exchange", pa.string()),
        ("symbol", pa.string()),
        ("bid_px", pa.list_(pa.float64())),
        ("bid_sz", pa.list_(pa.float64())),
        ("ask_px", pa.list_(pa.float64())),
        ("ask_sz", pa.list_(pa.float64())),
    ]
)

# (kind, date, exchange)
PartitionKey = tuple[str, str, str]


def _utc_date(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, UTC).strftime("%Y-%m-%d")


def _trade_row(t: Trade) -> dict:
    return {
        "ts": t.ts,
        "recv_ts": t.recv_ts,
        "exchange": t.exchange,
        "symbol": t.symbol,
        "price": t.price,
        "amount": t.amount,
        "side": t.side,
        "id": t.id,
    }


def _book_row(b: OrderBook) -> dict:
    return {
        "ts": b.ts,
        "recv_ts": b.recv_ts,
        "exchange": b.exchange,
        "symbol": b.symbol,
        "bid_px": [p for p, _ in b.bids],
        "bid_sz": [a for _, a in b.bids],
        "ask_px": [p for p, _ in b.asks],
        "ask_sz": [a for _, a in b.asks],
    }


class ParquetSink:
    def __init__(
        self, root: Path | str, flush_interval_s: float = 60.0, book_interval_ms: int = 1_000
    ) -> None:
        self.root = Path(root)
        self.flush_interval_s = flush_interval_s
        self.book_interval_ms = book_interval_ms
        self._buffer: dict[PartitionKey, list[dict]] = defaultdict(list)
        self._last_book_ts: dict[tuple[str, str], int] = {}
        self.rows_written = 0

    def add(self, event: MarketEvent) -> None:
        if isinstance(event, Trade):
            self._buffer["trades", _utc_date(event.ts), event.exchange].append(_trade_row(event))
            return
        key = (event.exchange, event.symbol)
        last = self._last_book_ts.get(key)
        if last is not None and event.ts - last < self.book_interval_ms:
            return
        self._last_book_ts[key] = event.ts
        self._buffer["books", _utc_date(event.ts), event.exchange].append(_book_row(event))

    async def run(self, queue: asyncio.Queue[MarketEvent]) -> None:
        """Consume the queue forever, flushing periodically and once more on cancellation."""
        loop = asyncio.get_running_loop()
        next_flush = loop.time() + self.flush_interval_s
        try:
            while True:
                timeout = max(0.0, next_flush - loop.time())
                try:
                    self.add(await asyncio.wait_for(queue.get(), timeout))
                except TimeoutError:
                    pass
                if loop.time() >= next_flush:
                    await self.flush()
                    next_flush = loop.time() + self.flush_interval_s
        finally:
            while not queue.empty():
                self.add(queue.get_nowait())
            # Final flush runs synchronously: the task is being cancelled and must not await.
            self._write(self._take_buffer())

    async def flush(self) -> None:
        await asyncio.to_thread(self._write, self._take_buffer())

    def _take_buffer(self) -> dict[PartitionKey, list[dict]]:
        buffer, self._buffer = self._buffer, defaultdict(list)
        return buffer

    def _write(self, buffer: dict[PartitionKey, list[dict]]) -> None:
        for (kind, date, exchange), rows in buffer.items():
            if not rows:
                continue
            schema = TRADE_SCHEMA if kind == "trades" else BOOK_SCHEMA
            directory = self.root / kind / f"date={date}" / f"exchange={exchange}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"part-{rows[0]['ts']}-{secrets.token_hex(4)}.parquet"
            table = pa.Table.from_pylist(rows, schema=schema).sort_by(
                [("symbol", "ascending"), ("ts", "ascending")]
            )
            write_parquet_atomic(table, path)
            self.rows_written += len(rows)
            log.debug("wrote %d %s rows to %s", len(rows), kind, path)
