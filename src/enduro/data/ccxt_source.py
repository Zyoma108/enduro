"""MarketDataSource implementation on top of ccxt (WebSocket streams + REST catalog/history)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

import ccxt
import ccxt.pro as ccxtpro

from enduro.core.models import Candle, OrderBook, Trade, now_ms

log = logging.getLogger(__name__)

# ccxt market "types" to load for each of our market types; loading everything (options
# in particular) costs dozens of extra requests on startup.
_MARKET_TYPES = {"swap": ["linear"], "spot": ["spot"]}


# Exchange errors that are server-side hiccups rather than bad requests.
_TRANSIENT_ERROR_CODES = ('"retCode":10016',)  # bybit: "svc error: Get kline failed"


def is_transient(error: Exception) -> bool:
    if isinstance(error, ccxt.NetworkError):
        return True
    return isinstance(error, ccxt.ExchangeError) and any(
        code in str(error) for code in _TRANSIENT_ERROR_CODES
    )


async def with_retries[T](
    call: Callable[[], Awaitable[T]], what: str, attempts: int = 4, delay_s: float = 1.0
) -> T:
    """Retry transient errors (timeouts, disconnects, server hiccups) with a growing delay."""
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except Exception as e:
            if attempt == attempts or not is_transient(e):
                raise
            log.warning("%s failed (%s), retry %d/%d", what, type(e).__name__, attempt, attempts)
            await asyncio.sleep(delay_s * attempt)
    raise AssertionError("unreachable")


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


def candle_from_ccxt(exchange: str, symbol: str, raw: list) -> Candle:
    ts, o, h, low, c, v = raw[:6]
    return Candle(
        exchange=exchange,
        symbol=symbol,
        ts=int(ts),
        open=float(o),
        high=float(h),
        low=float(low),
        close=float(c),
        volume=float(v or 0.0),
    )


def is_linear_usdt_perp(market: dict[str, Any]) -> bool:
    return bool(
        market.get("swap")
        and market.get("linear")
        and market.get("quote") == "USDT"
        and market.get("settle") == "USDT"
        and market.get("active")
    )


# Exchange-specific tags for non-crypto contracts (stocks, ETFs, commodities, FX, pre-IPO).
_CRYPTO_TAGS = {
    "bybit": ("symbolType", {"", "innovation"}),
    "binance": ("underlyingType", {"COIN"}),
}


def asset_class(exchange: str, market: dict[str, Any]) -> str:
    """'crypto' or 'tradfi'. Exchanges without known tags are assumed crypto-only."""
    if exchange not in _CRYPTO_TAGS:
        return "crypto"
    field, crypto_values = _CRYPTO_TAGS[exchange]
    value = (market.get("info") or {}).get(field)
    return "crypto" if value is None or value in crypto_values else "tradfi"


# WebSocket keepalive (ms): ping/pong interval; a connection missing two pongs is failed.
# ccxt's Binance default is 180 s, so a link silently dropped by the network was only
# noticed after up to 6 minutes of frozen data (seen live: AAVE book 118 s old, no error).
WS_KEEPALIVE_MS = 15_000

RECENT_TRADES_PAGE = 1000  # the most both exchanges return per request
RECENT_TRADES_MAX_PAGES = 10
# ccxt spaces Binance aggTrades calls ~1 s apart (request weight 20): a hot coin (~90
# prints/s) would need 80+ pages for 15 minutes. Stop paging after this long and keep
# the newest part, which matters most.
RECENT_TRADES_BUDGET_S = 5.0
# Without an API key Binance serves older trades only in aggregated form (one print per
# taker order and price), so a stream that is backfilled must be aggregated as well.
_RAW_TRADES_METHOD = {
    ("binance", "swap"): "fapiPublicGetTrades",
    ("binance", "spot"): "publicGetTrades",
}
_AGGREGATED_TRADES_METHOD = {
    ("binance", "swap"): "fapiPublicGetAggTrades",
    ("binance", "spot"): "publicGetAggTrades",
}


class CcxtSource:
    def __init__(
        self,
        exchange: str,
        market_type: str = "swap",
        book_limit: int | None = None,
        aggregated_trades: bool = False,
        **options: Any,
    ) -> None:
        """`book_limit` — order book depth to subscribe to (exchange-specific valid values,
        e.g. Bybit 1/50/200/1000; Binance always keeps a full local book).
        `aggregated_trades` — stream and fetch aggregated trades (Binance only): needed to
        backfill more than the last minute of trades."""
        try:
            client_cls = getattr(ccxtpro, exchange)
        except AttributeError:
            raise ValueError(f"ccxt has no WebSocket support for exchange {exchange!r}") from None
        if aggregated_trades:
            if (exchange, market_type) not in _AGGREGATED_TRADES_METHOD:
                raise ValueError(f"no aggregated trades on {exchange} {market_type}")
            options = {**options, "watchTradesForSymbols": {"name": "aggTrade"}}
            self._trades_method = _AGGREGATED_TRADES_METHOD[exchange, market_type]
        else:
            self._trades_method = _RAW_TRADES_METHOD.get((exchange, market_type))
        self._aggregated = aggregated_trades
        self.exchange = exchange
        self._book_limit = book_limit
        self._client = client_cls(
            {
                "enableRateLimit": True,
                "streaming": {"keepAlive": WS_KEEPALIVE_MS},
                "options": {
                    "defaultType": market_type,
                    "fetchMarkets": {"types": _MARKET_TYPES[market_type]},
                    **options,
                },
            }
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
            raw = await self._client.watch_order_book_for_symbols(list(symbols), self._book_limit)
            yield book_from_ccxt(self.exchange, raw, depth, now_ms())

    async def fetch_recent_trades(self, symbol: str, since: int) -> list[Trade]:
        """Trades after `since`, oldest first. Raw trades: only the latest page (Binance
        ~0.5-1 min of a busy coin, Bybit ~1-5 min). Aggregated: paged back by id until
        `since`, `RECENT_TRADES_MAX_PAGES` or `RECENT_TRADES_BUDGET_S`."""
        params = {"fetchTradesMethod": self._trades_method} if self._trades_method else {}

        async def page(extra: dict[str, Any]) -> list[dict[str, Any]]:
            return await with_retries(
                lambda: self._client.fetch_trades(
                    symbol, None, RECENT_TRADES_PAGE, {**params, **extra}
                ),
                f"{self.exchange} fetch_trades {symbol}",
            )

        deadline = asyncio.get_running_loop().time() + RECENT_TRADES_BUDGET_S
        raw = await page({})
        if self._aggregated:
            for _ in range(RECENT_TRADES_MAX_PAGES - 1):
                if not raw or raw[0]["timestamp"] <= since or int(raw[0]["id"]) == 0:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    break
                older = await page({"fromId": max(0, int(raw[0]["id"]) - RECENT_TRADES_PAGE)})
                older = [t for t in older if int(t["id"]) < int(raw[0]["id"])]
                if not older:
                    break
                raw = older + raw
        recv_ts = now_ms()
        trades = [trade_from_ccxt(self.exchange, r, recv_ts) for r in raw]
        return sorted((t for t in trades if t.ts > since), key=lambda t: t.ts)

    async def reset_streams(self) -> None:
        """Drop every WebSocket connection; the next watch call opens fresh ones."""
        await self._client.close_ws_clients()

    async def unsubscribe_trades(self, symbols: Sequence[str]) -> None:
        await self._client.un_watch_trades_for_symbols(list(symbols))

    async def unsubscribe_order_books(self, symbols: Sequence[str]) -> None:
        # The unsubscribe topic must name the same depth as the subscription (Bybit:
        # orderbook.<depth>.<symbol>); ccxt otherwise assumes its default (500 on Bybit
        # linear) and the real subscription stays alive on the server, so the next
        # subscribe fails with "already subscribed".
        params = {"limit": self._book_limit} if self._book_limit else {}
        await self._client.un_watch_order_book_for_symbols(list(symbols), params)

    async def list_linear_usdt_perps(self) -> dict[str, str]:
        markets = await with_retries(self._client.load_markets, f"{self.exchange} load_markets")
        return {
            m["symbol"]: asset_class(self.exchange, m)
            for m in markets.values()
            if is_linear_usdt_perp(m)
        }

    async def fetch_quote_volumes(self, symbols: Sequence[str]) -> dict[str, float]:
        """24h traded volume in quote currency (USDT) per symbol."""
        tickers = await with_retries(
            lambda: self._client.fetch_tickers(list(symbols)), f"{self.exchange} fetch_tickers"
        )
        return {s: float(t.get("quoteVolume") or 0.0) for s, t in tickers.items()}

    async def fetch_order_book(self, symbol: str, limit: int) -> OrderBook:
        """REST snapshot of the top `limit` levels per side."""
        raw = await with_retries(
            lambda: self._client.fetch_order_book(symbol, limit), f"{self.exchange} order book"
        )
        return book_from_ccxt(self.exchange, {**raw, "symbol": symbol}, limit, now_ms())

    async def fetch_top_of_book(self, symbol: str) -> OrderBook:
        return await self.fetch_order_book(symbol, 1)

    async def fetch_candles(
        self, symbol: str, since: int, limit: int = 1000, timeframe: str = "1m"
    ) -> list[Candle]:
        """Candles with open time >= `since`, oldest first. The last one may still be open."""
        raw = await with_retries(
            lambda: self._client.fetch_ohlcv(symbol, timeframe, since, limit),
            f"{self.exchange} fetch_ohlcv {symbol}",
        )
        return [candle_from_ccxt(self.exchange, symbol, r) for r in raw]

    async def close(self) -> None:
        await self._client.close()
