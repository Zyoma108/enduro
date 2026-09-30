import pytest

from enduro.analytics.market_state import MarketState
from enduro.core.models import OrderBook, Trade


def book(exchange: str, bid: float, ask: float) -> OrderBook:
    return OrderBook(exchange, "X", 0, 0, bids=((bid, 1.0),), asks=((ask, 1.0),))


def trade(ts: int, price: float, amount: float, side="buy") -> Trade:
    return Trade("binance", "X", ts, ts, price=price, amount=amount, side=side)


def test_trade_stats_over_window():
    state = MarketState(window_ms=1_000)
    state.on_event(trade(0, 100.0, 1.0, "buy"))  # falls out of the window
    state.on_event(trade(1_500, 100.0, 1.0, "buy"))
    state.on_event(trade(1_800, 200.0, 1.0, "sell"))

    stats = state.trade_stats("binance", "X", now_ms=2_000)
    assert stats.count == 2
    assert stats.volume == 2.0
    assert stats.notional == 300.0
    assert stats.vwap == 150.0
    assert stats.buy_ratio == pytest.approx(1 / 3)


def test_trade_stats_empty():
    stats = MarketState().trade_stats("binance", "X", now_ms=0)
    assert stats.count == 0
    assert stats.vwap is None
    assert stats.buy_ratio is None


def test_divergence_bps():
    state = MarketState()
    assert state.divergence_bps("X", "binance", "bybit") is None
    state.on_event(book("binance", 99.0, 101.0))  # mid 100
    state.on_event(book("bybit", 100.0, 102.0))  # mid 101
    assert state.divergence_bps("X", "binance", "bybit") == pytest.approx(100.0)
