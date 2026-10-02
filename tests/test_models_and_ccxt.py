import pytest

from enduro.core.models import OrderBook
from enduro.data.ccxt_source import book_from_ccxt, trade_from_ccxt


def test_trade_from_ccxt():
    raw = {
        "symbol": "BTC/USDT:USDT",
        "timestamp": 1_000,
        "price": "100.5",
        "amount": 2,
        "side": "sell",
        "id": 42,
    }
    trade = trade_from_ccxt("binance", raw, recv_ts=1_005)
    assert trade.exchange == "binance"
    assert trade.ts == 1_000
    assert trade.recv_ts == 1_005
    assert trade.price == 100.5
    assert trade.side == "sell"
    assert trade.id == "42"
    assert trade.notional == 201.0


def test_trade_from_ccxt_missing_fields():
    raw = {"symbol": "X", "timestamp": None, "price": 1, "amount": 1, "side": None}
    trade = trade_from_ccxt("bybit", raw, recv_ts=7)
    assert trade.ts == 7
    assert trade.side is None
    assert trade.id is None


def test_book_from_ccxt_truncates_and_copies():
    bids = [[100.0, 1.0], [99.0, 2.0], [98.0, 3.0]]
    raw = {"symbol": "X", "timestamp": 10, "bids": bids, "asks": [[101.0, 1.0, 0]]}
    book = book_from_ccxt("bybit", raw, depth=2, recv_ts=11)
    bids[0][0] = 0.0  # ccxt mutates its books in place; our snapshot must not change
    assert book.bids == ((100.0, 1.0), (99.0, 2.0))
    assert book.asks == ((101.0, 1.0),)


def test_order_book_metrics():
    book = OrderBook("x", "X", 0, 0, bids=((99.0, 1.0),), asks=((101.0, 1.0),))
    assert book.best_bid == 99.0
    assert book.best_ask == 101.0
    assert book.mid == 100.0
    assert book.spread_bps == pytest.approx(200.0)


def test_empty_order_book_metrics():
    book = OrderBook("x", "X", 0, 0, bids=(), asks=((101.0, 1.0),))
    assert book.mid is None
    assert book.spread_bps is None


class PagedTradesClient:
    """Serves aggregated trades with ids 0..n-1, one per second, newest page by default."""

    def __init__(self, n: int, t0: int) -> None:
        self.n, self.t0 = n, t0
        self.calls: list[dict] = []

    async def fetch_trades(self, symbol, since, limit, params):
        self.calls.append(params)
        start = params.get("fromId", max(0, self.n - limit))
        return [
            {
                "symbol": symbol,
                "id": str(i),
                "timestamp": self.t0 + i * 1000,
                "price": 1.0,
                "amount": 1.0,
                "side": "buy",
            }
            for i in range(start, min(start + limit, self.n))
        ]

    async def close(self):
        pass


async def test_aggregated_recent_trades_are_paged_back_to_since():
    from enduro.data.ccxt_source import CcxtSource

    src = CcxtSource("binance", aggregated_trades=True)
    await src._client.close()
    src._client = client = PagedTradesClient(n=3_500, t0=0)
    trades = await src.fetch_recent_trades("X", since=1_200_000)  # after trade id 1200
    assert [t.id for t in trades] == [str(i) for i in range(1201, 3500)]
    assert [c.get("fromId") for c in client.calls] == [None, 1500, 500]
    assert all(c["fetchTradesMethod"] == "fapiPublicGetAggTrades" for c in client.calls)


async def test_raw_recent_trades_are_one_page():
    from enduro.data.ccxt_source import CcxtSource

    src = CcxtSource("binance")
    await src._client.close()
    src._client = client = PagedTradesClient(n=3_500, t0=0)
    trades = await src.fetch_recent_trades("X", since=0)
    assert len(trades) == 1000 and client.calls == [{"fetchTradesMethod": "fapiPublicGetTrades"}]
    with pytest.raises(ValueError):
        CcxtSource("bybit", aggregated_trades=True)
