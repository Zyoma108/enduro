"""Normalized market data models shared by every layer.

These are hot-path objects (thousands per second), so they are plain slotted
dataclasses rather than pydantic models. All timestamps are Unix epoch
milliseconds: `ts` is the exchange's event time, `recv_ts` is when we received it
locally — the gap between them matters for cross-exchange lead/lag analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

Side = Literal["buy", "sell"]
PriceLevel = tuple[float, float]  # (price, amount)


def now_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass(frozen=True, slots=True)
class Trade:
    exchange: str
    symbol: str
    ts: int
    recv_ts: int
    price: float
    amount: float
    side: Side | None  # taker side; None if the exchange does not report it
    id: str | None = None

    @property
    def notional(self) -> float:
        return self.price * self.amount


@dataclass(frozen=True, slots=True)
class OrderBook:
    """Top-N snapshot of an order book. Bids descending, asks ascending."""

    exchange: str
    symbol: str
    ts: int
    recv_ts: int
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2

    @property
    def spread_bps(self) -> float | None:
        mid = self.mid
        if mid is None:
            return None
        return (self.asks[0][0] - self.bids[0][0]) / mid * 1e4


@dataclass(frozen=True, slots=True)
class Candle:
    """OHLCV bar; `ts` is the bar's open time."""

    exchange: str
    symbol: str
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float  # base currency


MarketEvent = Trade | OrderBook

MINUTE_MS = 60_000
