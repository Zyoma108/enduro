import pytest

from enduro.analytics import focus as f
from enduro.core.models import OrderBook, Trade

T0 = 1_790_726_400_000


def trade(ts, price, amount, side, exchange="binance", symbol="X"):
    return Trade(exchange, symbol, ts, ts, price=price, amount=amount, side=side)


def book(ts, bids, asks, exchange="bybit", symbol="X"):
    return OrderBook(exchange, symbol, ts, ts, bids=tuple(bids), asks=tuple(asks))


def test_flow_stats_window_delta_vwap_and_large_prints():
    trades = [
        trade(T0 - 1, 100, 99, "buy"),  # before the window: ignored
        trade(T0, 100, 1, "buy"),
        trade(T0 + 1_000, 102, 1, "sell"),
        trade(T0 + 2_000, 104, 2, "buy"),
    ]
    stats = f.flow_stats(trades, T0 - 1, T0 + 9_999, large_threshold=200, rate_15m=0.15)
    assert stats.trades == 3
    assert stats.notional == pytest.approx(100 + 102 + 208)
    assert stats.delta_ratio == pytest.approx((308 - 102) / 410)
    assert stats.intensity == pytest.approx(0.3)
    assert stats.intensity_vs_15m == pytest.approx(2.0)
    assert stats.price_change == pytest.approx(0.04)
    assert stats.vwap == pytest.approx(410 / 4)
    assert stats.price_vs_vwap_bps == pytest.approx((104 / 102.5 - 1) * 1e4)
    assert (stats.large_buy_notional, stats.large_sell_notional) == (208, 0)


def test_flow_stats_empty_window():
    stats = f.flow_stats([], T0, T0 + 60_000, None, 0.0)
    assert (stats.trades, stats.delta_ratio, stats.vwap, stats.price_vs_vwap_bps) == (
        0,
        0.0,
        None,
        None,
    )


def test_large_trade_threshold_needs_enough_trades():
    assert f.large_trade_threshold([trade(T0, 1, 1, "buy")] * 50) is None
    trades = [trade(T0, 1, 1, "buy")] * 199 + [trade(T0, 1, 1000, "sell")]
    # identical small lots: p99 is 1.0, so the median-multiple floor decides
    assert f.large_trade_threshold(trades) == 10.0
    varied = [trade(T0, 1, i % 50 + 1, "buy") for i in range(1000)]
    assert f.large_trade_threshold(varied) == pytest.approx(10 * 25.5)  # 10x median wins


def test_depth_and_slippage_walk_the_book():
    asks = [(100.0, 10.0), (100.2, 10.0), (101.0, 100.0)]
    assert f.depth_within(asks, mid=100.0, bps=25) == pytest.approx(1000 + 1002)
    # 1500 USDT: 1000 at 100.0, 500 at 100.2 -> avg = 1500 / (10 + 500/100.2)
    avg = 1500 / (10 + 500 / 100.2)
    assert f.slippage_bps(asks, 100.0, 1500) == pytest.approx((avg / 100 - 1) * 1e4)
    assert f.slippage_bps(asks, 100.0, 1e9) is None  # deeper than the visible book
    bids = [(99.9, 5.0)]
    assert f.slippage_bps(bids, 100.0, 100) == pytest.approx(10.0)


def test_book_stats():
    b = book(T0, bids=[(99.9, 10.0)], asks=[(100.1, 30.0)])
    stats = f.book_stats(b, spread_history=[10.0, 20.0, 40.0], notionals=[500.0])
    assert stats.spread_bps == pytest.approx(20.0)
    assert stats.spread_vs_15m == pytest.approx(1.0)
    assert stats.visible_bps == pytest.approx((10.0, 10.0))
    assert stats.depth[10] == pytest.approx((999.0, 3003.0))
    assert stats.imbalance(10) == pytest.approx((999 - 3003) / (999 + 3003))
    assert stats.depth[25] == (None, None)  # the book we received does not reach 25 bps
    assert stats.imbalance(25) is None
    assert stats.slippage[500.0][0] == pytest.approx(10.0)
    assert f.book_stats(book(T0, [], [(1.0, 1.0)]), [], [1.0]) is None


def test_basis_series_aligns_by_second():
    ref = [(T0, 100.0), (T0 + 1_000, 100.0)]
    exe = [(T0 + 300, 100.1), (T0 + 2_500, 100.2)]  # second sample has no reference
    assert f.basis_series(ref, exe) == pytest.approx([10.0])


def test_confirms():
    up_buy = f.flow_stats(
        [trade(T0, 100, 1, "buy"), trade(T0 + 1, 101, 1, "buy")], T0 - 1, T0 + 9, None, 1
    )
    down_sell = f.flow_stats(
        [trade(T0, 101, 1, "sell"), trade(T0 + 1, 100, 1, "sell")], T0 - 1, T0 + 9, None, 1
    )
    flat = f.flow_stats([trade(T0, 100, 1, "buy")], T0 - 1, T0 + 9, None, 1)
    assert f.confirms(up_buy, up_buy) is True
    assert f.confirms(up_buy, down_sell) is False
    assert f.confirms(up_buy, flat) is None
    # a 0.1 bp wiggle is noise, not a direction
    wiggle = f.flow_stats(
        [trade(T0, 100, 1, "sell"), trade(T0 + 1, 99.999, 1, "sell")], T0 - 1, T0 + 9, None, 1
    )
    assert f.confirms(up_buy, wiggle) is None


def test_tracker_snapshot_end_to_end():
    tracker = f.FocusTracker("binance", "bybit", slippage_notionals=[1_000])
    for i in range(120):  # two minutes: binance leads up with buyers, bybit follows
        ts = T0 + i * 1_000
        tracker.on_event(trade(ts, 100 + i * 0.01, 1, "buy", "binance"))
        tracker.on_event(trade(ts, 100 + i * 0.01, 0.5, "buy", "bybit"))
        mid = 100 + i * 0.01
        tracker.on_event(book(ts, [(mid - 0.01, 50)], [(mid + 0.01, 50)], "binance"))
        tracker.on_event(book(ts, [(mid - 0.005, 50)], [(mid + 0.015, 50)], "bybit"))
    tracker.on_event(trade(T0, 1, 1, "buy", "binance", symbol="OTHER"))

    snap = tracker.snapshot("X", now_ms=T0 + 120_000)
    assert snap.observed_s == pytest.approx(120)
    one_min = snap.flow["binance"]["1m"]
    # window is (now - 60s, now]: trades at +61s..+119s
    assert one_min.trades == 59
    assert one_min.delta_ratio == 1.0
    assert one_min.intensity_vs_15m == pytest.approx((59 / 60) / (120 / 120))
    assert snap.book["bybit"].spread_bps == pytest.approx(0.02 / snap.book["bybit"].mid * 1e4)
    assert snap.cross.basis_bps == pytest.approx(0.005 / snap.book["binance"].mid * 1e4)
    assert snap.cross.basis_std_bps == pytest.approx(0.0, abs=0.01)  # 0.005 abs on a rising mid
    assert snap.cross.reference_volume_share["1m"] == pytest.approx(2 / 3, rel=1e-3)
    assert snap.cross.confirms["5m"] is True

    summary = snap.to_summary()
    assert list(summary["flow"]["bybit"]) == ["1m", "5m"]  # 15m == 5m after 2 minutes
    assert "partial_window_s" not in summary["flow"]["bybit"]["1m"]
    assert summary["flow"]["bybit"]["5m"]["partial_window_s"] == 120
    assert summary["flow"]["bybit"]["1m"]["delta_ratio"] == 1.0
    assert summary["cross"]["binance_confirms"]["1m"] is True
    assert "1000usd" in summary["book"]["bybit"]["slippage_bps"]

    tracker.drop("X")
    assert tracker.snapshot("X", T0 + 120_000) is None
    assert tracker.snapshot("OTHER", T0 + 120_000) is not None


def test_tracker_evicts_old_history():
    tracker = f.FocusTracker("binance", "bybit")
    tracker.on_event(trade(T0, 100, 1, "buy"))
    tracker.on_event(trade(T0 + f.HISTORY_MS + 5_000, 100, 1, "buy"))
    snap = tracker.snapshot("X", T0 + f.HISTORY_MS + 5_001)
    assert snap.flow["binance"]["15m"].trades == 1


def test_windows_do_not_extend_before_observation_start():
    tracker = f.FocusTracker("binance", "bybit")
    for i in range(30):  # watched for 30 s at 2 trades/s
        tracker.on_event(trade(T0 + i * 500, 100, 1, "buy"))
        tracker.on_event(trade(T0 + i * 500 + 250, 100, 1, "buy"))
    snap = tracker.snapshot("X", T0 + 30_000)
    five = snap.flow["binance"]["5m"]
    assert five.seconds == pytest.approx(30.001)
    assert five.intensity == pytest.approx(2.0, rel=1e-3)  # not diluted over 300 s
    assert five.intensity_vs_15m == pytest.approx(1.0, rel=1e-3)
    flow = snap.to_summary()["flow"]["binance"]
    assert list(flow) == ["1m"]  # 5m and 15m would repeat the same 30 s
    assert flow["1m"]["partial_window_s"] == 30
