"""Keeps a Radar fed with fresh 1m candles and persists them, for `enduro scan` and the agent."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from enduro.analytics.baseline import load_baselines
from enduro.analytics.liquidity import LiquidityBook, scan_liquidity
from enduro.analytics.radar import LOOKBACK_MINUTES, Radar, RadarRow
from enduro.core.models import MINUTE_MS, Candle, now_ms
from enduro.data.base import MarketDataSource
from enduro.data.history import backfill
from enduro.storage.candles import last_candle_ts, load_recent_candles, write_candles

log = logging.getLogger(__name__)

PERSIST_EVERY_MS = 60 * MINUTE_MS
LIQUIDITY_EVERY_MS = 5 * MINUTE_MS


class RadarRunner:
    def __init__(
        self,
        source: MarketDataSource,
        symbols: Sequence[str],
        root: Path,
        history_days: int,
        taker_fee_bps: float,
        liquidity_source: MarketDataSource | None = None,
        min_depth_usd: float = 2_000.0,
        max_spread_bps: float = 10.0,
    ) -> None:
        self.source = source
        self.symbols = list(symbols)
        self.root = root
        self.history_days = history_days
        self.round_trip_fee = 2 * taker_fee_bps / 1e4
        self.radar: Radar | None = None
        self.rows: list[RadarRow] = []
        self.updated_ms = 0
        # Execution-exchange order books: is the coin actually tradable there?
        self.liquidity_source = liquidity_source
        self.liquidity = LiquidityBook(min_depth_usd, max_spread_bps)
        self._pending: list[Candle] = []
        self._last_persist = now_ms()

    async def prepare(self) -> None:
        """Catch stored history up, load baselines, seed the radar's rolling window."""
        now = now_ms()
        root, exchange = self.root, self.source.exchange
        # Persisting continues stored history only if it has no gap, so catch up first.
        last = await asyncio.to_thread(last_candle_ts, root, exchange)
        start = now - self.history_days * 86_400_000

        async def persist(symbol: str, candles: list[Candle]) -> None:
            await asyncio.to_thread(write_candles, root, candles)

        await backfill(self.source, self.symbols, last, start, now, persist)
        baselines = await asyncio.to_thread(load_baselines, root, exchange, start)
        if missing := [s for s in self.symbols if s not in baselines]:
            log.warning("no baseline (history) for %d symbols: %s", len(missing), missing)
        self.radar = Radar(baselines, round_trip_fee=self.round_trip_fee)
        since = now - LOOKBACK_MINUTES * MINUTE_MS
        self.radar.add(
            await asyncio.to_thread(load_recent_candles, root, exchange, self.symbols, since)
        )

    async def refresh(self) -> list[RadarRow]:
        """Fetch candles closed since the last refresh and re-rank."""
        assert self.radar is not None, "call prepare() first"
        radar = self.radar
        now = now_ms()

        def collect(symbol: str, candles: list[Candle]) -> None:
            radar.add(candles)
            self._pending.extend(candles)

        await backfill(
            self.source,
            self.symbols,
            radar.last_ts(),
            now - LOOKBACK_MINUTES * MINUTE_MS,
            now,
            collect,
        )
        if self.liquidity_source and now - self.liquidity.updated_ms >= LIQUIDITY_EVERY_MS:
            snapshots = await scan_liquidity(self.liquidity_source, self.symbols)
            self.liquidity.add(snapshots, now)
        self.rows = radar.scan(self.symbols)
        self.updated_ms = now
        if now - self._last_persist >= PERSIST_EVERY_MS:
            await asyncio.to_thread(self.flush)
            self._last_persist = now
        return self.rows

    async def run_forever(self, on_update: Callable[[list[RadarRow]], None] | None = None) -> None:
        while True:
            rows = await self.refresh()
            if on_update:
                on_update(rows)
            # Wake a few seconds after the next minute closes, when exchanges have the candle.
            next_minute = (now_ms() // MINUTE_MS + 1) * MINUTE_MS
            await asyncio.sleep((next_minute - now_ms()) / 1000 + 3)

    def flush(self) -> None:
        if self._pending:
            batch = self._pending.copy()
            self._pending.clear()
            write_candles(self.root, batch)
