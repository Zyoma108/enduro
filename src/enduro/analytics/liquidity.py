"""Tradability of a symbol on the execution exchange: spread and resting depth near mid.

Daily volume alone is not enough: some coins trade $10M+ a day on Bybit yet show an
almost empty book within ±10 bps (seen live: MOVR, US). Positions there pay the
spread and slippage on every entry and exit, which eats intraday edge.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from enduro.analytics.focus import depth_within
from enduro.core.models import OrderBook
from enduro.data.base import MarketDataSource

log = logging.getLogger(__name__)

BAND_BPS = 10
BOOK_LEVELS = 200


@dataclass(frozen=True, slots=True)
class Liquidity:
    spread_bps: float
    depth_bid_usd: float  # resting within BAND_BPS of mid (or within the visible book)
    depth_ask_usd: float
    ts: int

    @property
    def depth_usd(self) -> float:
        """The thinner side: what a market order in the worse direction can rely on."""
        return min(self.depth_bid_usd, self.depth_ask_usd)

    def tradable(self, min_depth_usd: float, max_spread_bps: float) -> bool:
        return self.spread_bps <= max_spread_bps and self.depth_usd >= min_depth_usd


def liquidity_from_book(book: OrderBook, band_bps: float = BAND_BPS) -> Liquidity | None:
    mid, spread = book.mid, book.spread_bps
    if mid is None or spread is None:
        return None
    # If the snapshot reaches less than band_bps (deep books with tiny ticks, e.g. BTC),
    # this undercounts: fine for a minimum-depth check, the real depth is only larger.
    return Liquidity(
        spread_bps=spread,
        depth_bid_usd=depth_within(book.bids, mid, band_bps),
        depth_ask_usd=depth_within(book.asks, mid, band_bps),
        ts=book.recv_ts,
    )


async def scan_liquidity(
    source: MarketDataSource, symbols: Sequence[str], concurrency: int = 5
) -> dict[str, Liquidity]:
    semaphore = asyncio.Semaphore(concurrency)
    result: dict[str, Liquidity] = {}

    async def one(symbol: str) -> None:
        async with semaphore:
            try:
                book = await source.fetch_order_book(
                    symbol, BOOK_LIMIT_FOR.get(source.exchange, 50)
                )
            except Exception as e:
                log.warning("%s: order book for %s failed: %s", source.exchange, symbol, e)
                return
        if (liq := liquidity_from_book(book)) is not None:
            result[symbol] = liq

    await asyncio.gather(*(one(s) for s in symbols))
    return result


# REST depth limits differ per exchange; Bybit linear accepts up to 500.
BOOK_LIMIT_FOR = {"bybit": BOOK_LEVELS}


class LiquidityBook:
    """Recent liquidity snapshots per symbol; judged on the median of the last few, so a
    momentary thin book does not flip a coin to untradable (or the other way round)."""

    def __init__(self, min_depth_usd: float, max_spread_bps: float, keep: int = 3) -> None:
        self.min_depth_usd = min_depth_usd
        self.max_spread_bps = max_spread_bps
        self._history: dict[str, deque[Liquidity]] = {}
        self._keep = keep
        self.updated_ms = 0

    def add(self, snapshots: dict[str, Liquidity], now_ms: int) -> None:
        for symbol, liq in snapshots.items():
            self._history.setdefault(symbol, deque(maxlen=self._keep)).append(liq)
        self.updated_ms = now_ms

    def typical(self, symbol: str) -> Liquidity | None:
        history = self._history.get(symbol)
        if not history:
            return None
        return Liquidity(
            spread_bps=statistics.median(h.spread_bps for h in history),
            depth_bid_usd=statistics.median(h.depth_bid_usd for h in history),
            depth_ask_usd=statistics.median(h.depth_ask_usd for h in history),
            ts=history[-1].ts,
        )

    def tradable(self, symbol: str) -> bool | None:
        """None when we have no snapshot yet (unknown is not the same as illiquid)."""
        liq = self.typical(symbol)
        return None if liq is None else liq.tradable(self.min_depth_usd, self.max_spread_bps)

    def summary(self, symbol: str) -> dict[str, float | bool | None]:
        liq = self.typical(symbol)
        if liq is None:
            return {"bybit_liquidity": None}
        return {
            "bybit_spread_bps": round(liq.spread_bps, 2),
            "bybit_depth_10bps_usd": round(liq.depth_usd),
            "tradable": liq.tradable(self.min_depth_usd, self.max_spread_bps),
        }
