"""Exchange-agnostic market data interface.

Everything above the data layer depends on this protocol, never on ccxt directly,
so an implementation can be swapped (e.g. for a native SDK) without touching
analytics or the agent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from enduro.core.models import OrderBook, Trade


class MarketDataSource(Protocol):
    exchange: str

    def stream_trades(self, symbols: Sequence[str]) -> AsyncIterator[Trade]:
        """Yield public trades for the given symbols until cancelled."""
        ...

    def stream_order_books(self, symbols: Sequence[str], depth: int) -> AsyncIterator[OrderBook]:
        """Yield a top-`depth` snapshot every time a book for one of the symbols changes."""
        ...

    async def close(self) -> None: ...
