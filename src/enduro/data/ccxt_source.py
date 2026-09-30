"""MarketDataSource implementation on top of ccxt's WebSocket (formerly ccxt.pro) API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import ccxt.pro as ccxtpro

from enduro.core.models import OrderBook, Trade, now_ms


def trade_from_ccxt(exchange: str, raw: dict[str, Any], recv_ts: int) -> Trade:
    side = raw.get("side")
    return Trade(
        exchange=exchange,
        symbol=raw["symbol"],
        ts=raw.get("timestamp") or recv_ts,
        recv_ts=recv_ts,
        price=float(raw["price"]),
        amount=float(raw["amount"]),
        side=side if side in ("buy", "sell") else None,
        id=str(raw["id"]) if raw.get("id") is not None else None,
    )


def book_from_ccxt(exchange: str, raw: dict[str, Any], depth: int, recv_ts: int) -> OrderBook:
    # ccxt mutates its order book objects in place, so copy the levels we keep.
    return OrderBook(
        exchange=exchange,
        symbol=raw["symbol"],
        ts=raw.get("timestamp") or recv_ts,
        recv_ts=recv_ts,
        bids=tuple((float(p), float(a)) for p, a, *_ in raw["bids"][:depth]),
        asks=tuple((float(p), float(a)) for p, a, *_ in raw["asks"][:depth]),
    )


class CcxtSource:
    def __init__(self, exchange: str, market_type: str = "swap", **options: Any) -> None:
        try:
            client_cls = getattr(ccxtpro, exchange)
        except AttributeError:
            raise ValueError(f"ccxt has no WebSocket support for exchange {exchange!r}") from None
        self.exchange = exchange
        self._client = client_cls(
            {"enableRateLimit": True, "options": {"defaultType": market_type, **options}}
        )

    async def stream_trades(self, symbols: Sequence[str]) -> AsyncIterator[Trade]:
        while True:
            # ccxt returns only trades that arrived since the previous call (newUpdates).
            raw_trades = await self._client.watch_trades_for_symbols(list(symbols))
            recv_ts = now_ms()
            for raw in raw_trades:
                yield trade_from_ccxt(self.exchange, raw, recv_ts)

    async def stream_order_books(
        self, symbols: Sequence[str], depth: int
    ) -> AsyncIterator[OrderBook]:
        while True:
            raw = await self._client.watch_order_book_for_symbols(list(symbols))
            yield book_from_ccxt(self.exchange, raw, depth, now_ms())

    async def close(self) -> None:
        await self._client.close()
