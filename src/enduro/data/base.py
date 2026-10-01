"""Exchange-agnostic market data interface.

Everything above the data layer depends on this protocol, never on ccxt directly,
so an implementation can be swapped (e.g. for a native SDK) without touching
analytics or the agent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from enduro.core.models import Candle, OrderBook, Trade


class MarketDataSource(Protocol):
    exchange: str

    def stream_trades(self, symbols: Sequence[str]) -> AsyncIterator[Trade]:
        """Yield public trades for the given symbols until cancelled."""
        ...

    def stream_order_books(self, symbols: Sequence[str], depth: int) -> AsyncIterator[OrderBook]:
        """Yield a top-`depth` snapshot every time a book for one of the symbols changes."""
        ...

    async def unsubscribe_trades(self, symbols: Sequence[str]) -> None: ...

    async def unsubscribe_order_books(self, symbols: Sequence[str]) -> None: ...

    async def reset_streams(self) -> None:
        """Drop all streaming connections (e.g. one that went silent without an error)."""
        ...

    async def list_linear_usdt_perps(self) -> dict[str, str]:
        """Active USDT-margined perpetuals: unified symbol → asset class ('crypto' | 'tradfi')."""
        ...

    async def fetch_quote_volumes(self, symbols: Sequence[str]) -> dict[str, float]: ...

    async def fetch_order_book(self, symbol: str, limit: int) -> OrderBook: ...

    async def fetch_candles(
        self, symbol: str, since: int, limit: int = 1000, timeframe: str = "1m"
    ) -> list[Candle]: ...

    async def close(self) -> None: ...
