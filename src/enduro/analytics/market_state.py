"""Live view of the market built from the event stream: latest books and rolling trade stats."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from enduro.core.models import MarketEvent, OrderBook, Trade

Key = tuple[str, str]  # (exchange, symbol)


@dataclass(frozen=True, slots=True)
class TradeStats:
    count: int
    volume: float  # base currency
    notional: float  # quote currency
    buy_ratio: float | None  # share of taker-buy notional; None if sides unknown
    vwap: float | None


class MarketState:
    def __init__(self, window_ms: int = 60_000) -> None:
        self.window_ms = window_ms
        self._books: dict[Key, OrderBook] = {}
        self._trades: dict[Key, deque[Trade]] = {}

    def on_event(self, event: MarketEvent) -> None:
        key = (event.exchange, event.symbol)
        if isinstance(event, OrderBook):
            self._books[key] = event
        else:
            trades = self._trades.setdefault(key, deque())
            trades.append(event)
            self._evict(trades, event.ts)

    def book(self, exchange: str, symbol: str) -> OrderBook | None:
        return self._books.get((exchange, symbol))

    def trade_stats(self, exchange: str, symbol: str, now_ms: int) -> TradeStats:
        trades = self._trades.get((exchange, symbol), deque())
        self._evict(trades, now_ms)
        volume = sum(t.amount for t in trades)
        notional = sum(t.notional for t in trades)
        sided = [t for t in trades if t.side is not None]
        sided_notional = sum(t.notional for t in sided)
        buy_notional = sum(t.notional for t in sided if t.side == "buy")
        return TradeStats(
            count=len(trades),
            volume=volume,
            notional=notional,
            buy_ratio=buy_notional / sided_notional if sided_notional else None,
            vwap=notional / volume if volume else None,
        )

    def divergence_bps(self, symbol: str, reference: str, other: str) -> float | None:
        """How far `other`'s mid price is from `reference`'s, in basis points."""
        ref, oth = self.book(reference, symbol), self.book(other, symbol)
        if ref is None or oth is None or ref.mid is None or oth.mid is None:
            return None
        return (oth.mid - ref.mid) / ref.mid * 1e4

    def _evict(self, trades: deque[Trade], now_ms: int) -> None:
        cutoff = now_ms - self.window_ms
        while trades and trades[0].ts < cutoff:
            trades.popleft()
