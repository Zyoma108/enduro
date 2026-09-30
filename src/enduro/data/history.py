"""Candle history download: initial backfill and incremental catch-up share one code path."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence

from enduro.core.models import MINUTE_MS, Candle
from enduro.data.base import MarketDataSource

log = logging.getLogger(__name__)

PAGE_LIMIT = 1000  # both Binance and Bybit return at most 1000 candles per request


async def fetch_closed_candles(
    source: MarketDataSource, symbol: str, since: int, now_ms: int
) -> list[Candle]:
    """All closed 1m candles from the one containing `since` up to `now`, paging forward.

    On a failed request, returns the pages fetched so far (logging the error): callers
    persist them and the next run resumes after the last stored candle.
    """
    since = since // MINUTE_MS * MINUTE_MS
    candles: list[Candle] = []
    while since + MINUTE_MS <= now_ms:
        try:
            page = await source.fetch_candles(symbol, since, PAGE_LIMIT)
        except Exception:
            log.exception(
                "%s: candles for %s stopped at %d (%d fetched)",
                source.exchange,
                symbol,
                since,
                len(candles),
            )
            break
        page = [c for c in page if c.ts >= since and c.ts + MINUTE_MS <= now_ms]
        if not page:
            break
        candles.extend(page)
        since = page[-1].ts + MINUTE_MS
    return candles


async def backfill(
    source: MarketDataSource,
    symbols: Sequence[str],
    last_ts: dict[str, int],
    start_ms: int,
    now_ms: int,
    on_candles: Callable[[str, list[Candle]], Awaitable[None] | None],
    concurrency: int = 4,
) -> int:
    """Fetch candles for every symbol from after its last stored candle (or `start_ms`).

    `on_candles` is called once per symbol as soon as its candles are ready, so the
    caller can persist them without holding the whole history in memory.
    Returns the total number of candles fetched.
    """
    semaphore = asyncio.Semaphore(concurrency)
    total = 0

    async def one(symbol: str) -> None:
        nonlocal total
        since = max(start_ms, last_ts.get(symbol, -MINUTE_MS) + MINUTE_MS)
        async with semaphore:
            candles = await fetch_closed_candles(source, symbol, since, now_ms)
        if candles:
            result = on_candles(symbol, candles)
            if result is not None:
                await result
            total += len(candles)

    await asyncio.gather(*(one(s) for s in symbols))
    return total
