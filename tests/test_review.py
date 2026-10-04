from enduro.agent.review import TradeContext, review_trade, trade_context
from enduro.core.models import MINUTE_MS, Candle
from enduro.execution.models import ClosedTrade

T0 = 1_790_726_400_000  # a minute boundary


def bar(minute, low, high, close):
    return Candle("bybit", "X", T0 + minute * MINUTE_MS, close, high, low, close, 1.0)


def test_long_exited_early_then_stopped_out():
    # Open at minute 0 at 100, exit by hand at minute 3 at 99.5; stop 98, take 103.
    candles = [
        bar(0, 99.8, 101.0, 100.5),  # best +1%
        bar(1, 99.0, 100.6, 99.4),  # worst -1%
        bar(2, 99.3, 99.9, 99.6),
        *[bar(m, 98.5, 99.6, 99.0) for m in range(3, 10)],
        bar(10, 97.9, 99.0, 98.2),  # stop hit 7 min after the exit
        *[bar(m, 97.0, 98.5, 97.5) for m in range(11, 70)],
    ]
    exit_ms = T0 + 3 * MINUTE_MS
    review, final = review_trade(
        side="long",
        entry=100.0,
        exit_price=99.5,
        opened_ms=T0,
        closed_ms=exit_ms,
        stop=98.0,
        take=103.0,
        by_agent=True,
        candles=candles,
        now_ms=T0 + 70 * MINUTE_MS,
    )
    assert review["held_min"] == 3.0
    assert review["best_while_open_pct"] == 1.0
    assert review["worst_while_open_pct"] == -1.0
    assert review["after_exit_pct"] == {"15m": -2.01, "30m": -2.01, "60m": -2.01}
    assert review["if_held"] == "stop 98 would have been hit 8 min after your exit"
    assert final


def test_short_signs_and_take_profit_first_and_not_final_yet():
    candles = [bar(0, 99.5, 100.2, 99.8), bar(1, 98.0, 99.9, 98.5), bar(2, 96.5, 98.6, 97.0)]
    review, final = review_trade(
        side="short",
        entry=100.0,
        exit_price=99.8,
        opened_ms=T0,
        closed_ms=T0 + MINUTE_MS,
        stop=101.0,
        take=97.0,
        by_agent=True,
        candles=candles,
        now_ms=T0 + 3 * MINUTE_MS,
    )
    assert review["best_while_open_pct"] == 0.5 and review["worst_while_open_pct"] == -0.2
    assert review["after_exit_pct"]["15m"] is None  # not there yet
    assert review["if_held"] == "take profit 97 would have been hit 2 min after your exit"
    assert not final


def test_bar_touching_both_levels_counts_as_stop_and_exchange_exits_skip_if_held():
    candles = [bar(0, 99, 101, 100), bar(1, 95, 105, 100)]
    kwargs = dict(
        side="long",
        entry=100.0,
        exit_price=100.0,
        opened_ms=T0,
        closed_ms=T0 + MINUTE_MS,
        stop=96.0,
        take=104.0,
        candles=candles,
        now_ms=T0 + 2 * MINUTE_MS,
    )
    assert review_trade(by_agent=True, **kwargs)[0]["if_held"].startswith("stop 96")
    assert "if_held" not in review_trade(by_agent=False, **kwargs)[0]


def test_trade_context_finds_open_thesis_and_moved_stop():
    trade = ClosedTrade("c-2", "X", "long", 1.0, 100.0, 101.0, 1.0, 0.1, T0 + 10 * MINUTE_MS)
    risk = lambda ts, thesis: {  # noqa: E731
        "ts": ts,
        "kind": "risk",
        "thesis": thesis,
        "intent": {"symbol": "X", "side": "long"},
        "decision": {"approved": True},
    }
    opened = lambda ts, stop: {  # noqa: E731
        "ts": ts,
        "kind": "order",
        "action": "open",
        "result": {"ts": ts + 5},
        "request": {
            "symbol": "X",
            "position_side": "long",
            "stop_loss": stop,
            "take_profit": 105.0,
        },
    }
    orders = [opened(T0 - 100 * MINUTE_MS, 90.0), opened(T0, 98.0)]
    risks = [
        risk(T0 - 101 * MINUTE_MS, "old trade"),
        risk(T0 - 1, "breakout of 99.5"),
        {
            "ts": T0 + MINUTE_MS,
            "kind": "risk",
            "ok": True,
            "protection": {"stop_loss": 99.0, "take_profit": None},
            "position": {"symbol": "X", "side": "long"},
        },
        {
            "ts": T0 + 20 * MINUTE_MS,
            "kind": "risk",
            "ok": True,  # after the close
            "protection": {"stop_loss": 100.5},
            "position": {"symbol": "X", "side": "long"},
        },
    ]
    assert trade_context(trade, orders, risks) == TradeContext(
        opened_ms=T0 + 5, thesis="breakout of 99.5", stop=99.0, take=105.0
    )
    assert trade_context(trade, [], risks) == TradeContext()
