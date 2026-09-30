import math

import numpy as np
import pytest

from enduro.analytics import metrics as m
from enduro.analytics.radar import Radar, analyze
from enduro.core.models import MINUTE_MS, Candle

T0 = 1_790_726_400_000  # 2026-09-30 00:00 UTC


def candles(symbol: str, n: int, spread: float, volume: float, drift: float = 0.0):
    """n 1m candles; each bar spans `spread` (fraction) and moves `drift` (fraction)."""
    out, price = [], 100.0
    for i in range(n):
        close = price * (1 + drift)
        high = max(price, close) * (1 + spread / 2)
        low = min(price, close) * (1 - spread / 2)
        out.append(Candle("binance", symbol, T0 + i * MINUTE_MS, price, high, low, close, volume))
        price = close
    return out


def baseline_for(spread: float, quote_volume: float) -> m.HourlyBaseline:
    bars = m.Bars.from_candles(candles("B", 60, spread, 1.0))
    variance = float(np.mean(m.gk_variance(bars)))
    return m.HourlyBaseline(np.full(24, variance), np.full(24, quote_volume))


def test_analyze_normal_market_is_about_one():
    bars = m.Bars.from_candles(candles("X", 240, spread=0.001, volume=1.0))
    row = analyze("X", bars, baseline_for(0.001, 100.0), round_trip_fee=0.0011)
    w = row.windows["15m"]
    assert w.vol_ratio == pytest.approx(1.0, rel=1e-6)
    assert w.volume_ratio == pytest.approx(1.0)
    assert row.score == pytest.approx(1.0, rel=1e-6)
    assert row.expansion == pytest.approx(1.0)
    assert w.vol_vs_24h == pytest.approx(1.0)
    assert w.volume_vs_24h == pytest.approx(1.0)
    assert row.day_volume_ratio == pytest.approx(1.0)
    assert row.move_vs_fees == pytest.approx(w.expected_move / 0.0011)
    assert row.price == 100.0


def test_analyze_detects_hot_trending_symbol():
    history = candles("X", 200, spread=0.001, volume=1.0)
    hot = [
        Candle(c.exchange, c.symbol, T0 + (200 + i) * MINUTE_MS, c.open, c.high, c.low, c.close, 4)
        for i, c in enumerate(candles("X", 40, spread=0.003, volume=4.0, drift=0.002))
    ]
    bars = m.Bars.from_candles(history + hot)
    row = analyze("X", bars, baseline_for(0.001, 100.0), round_trip_fee=0.0011)
    w = row.windows["15m"]
    assert w.vol_ratio > 3
    assert w.volume_ratio > 3
    assert w.change > 0.02
    assert w.efficiency > 0.9
    assert row.expansion > 1
    assert w.volume_vs_24h > 2  # hotter than its own last day
    assert w.vol_vs_24h > 1.5


def test_symbol_hot_all_day_is_in_play_but_not_heating_up():
    bars = m.Bars.from_candles(candles("X", 1440, spread=0.004, volume=5.0))
    row = analyze("X", bars, baseline_for(0.001, 100.0), round_trip_fee=0.0011)
    w = row.windows["15m"]
    assert row.day_volume_ratio == pytest.approx(5.0)  # in play vs the long-term norm
    assert w.volume_ratio == pytest.approx(5.0)
    assert w.volume_vs_24h == pytest.approx(1.0)  # ...but no hotter than the rest of the day
    assert w.vol_vs_24h == pytest.approx(1.0)


def test_analyze_needs_an_hour_of_data():
    bars = m.Bars.from_candles(candles("X", 59, spread=0.001, volume=1.0))
    assert analyze("X", bars, baseline_for(0.001, 100.0), 0.0011) is None


def test_to_summary_is_rounded_and_nan_safe():
    bars = m.Bars.from_candles(candles("X", 60, spread=0.001, volume=1.0))
    row = analyze("X", bars, baseline_for(0.001, 100.0), 0.0011)
    summary = row.to_summary()
    assert summary["15m"]["vol_vs_usual"] == 1.0
    assert summary["atr_5m_pct"] is None  # only 12 5m bars, ATR(14) needs 15


def test_radar_ranks_and_ignores_duplicates_and_unknown_symbols():
    radar = Radar({s: baseline_for(0.001, 100.0) for s in ("CALM", "HOT")}, round_trip_fee=0.0011)
    radar.add(candles("CALM", 120, spread=0.001, volume=1.0))
    radar.add(candles("HOT", 120, spread=0.004, volume=5.0))
    radar.add(candles("NOBASELINE", 120, spread=0.01, volume=9.0))
    radar.add(candles("CALM", 5, spread=0.5, volume=1e6))  # stale duplicates: ignored

    rows = radar.scan()
    assert [r.symbol for r in rows] == ["HOT", "CALM"]
    assert rows[1].score == pytest.approx(1.0, rel=1e-6)
    assert radar.last_ts()["CALM"] == T0 + 119 * MINUTE_MS
    assert [r.symbol for r in radar.scan(["CALM"])] == ["CALM"]
    assert not math.isnan(rows[0].score)
