import math

import numpy as np
import pytest

from enduro.analytics import metrics as m
from enduro.analytics.baseline import load_baselines
from enduro.core.models import MINUTE_MS, Candle
from enduro.storage.candles import write_candles

T0 = 1_790_726_400_000  # 2026-09-30 00:00 UTC


def bars_from(rows, start=T0) -> m.Bars:
    """rows: (open, high, low, close[, volume]) per minute."""
    return m.Bars.from_candles(
        [
            Candle("x", "X", start + i * MINUTE_MS, o, h, lo, c, r[4] if len(r) > 4 else 1.0)
            for i, r in enumerate(rows)
            for o, h, lo, c in [r[:4]]
        ]
    )


def test_gk_variance_matches_formula_and_is_zero_for_flat_bar():
    bars = bars_from([(100, 102, 99, 101), (100, 100, 100, 100)])
    hl, co = math.log(102 / 99), math.log(101 / 100)
    expected = 0.5 * hl**2 - (2 * math.log(2) - 1) * co**2
    assert m.gk_variance(bars) == pytest.approx([expected, 0.0])


def test_gk_estimates_random_walk_volatility():
    rng = np.random.default_rng(7)
    sigma_minute = 0.001
    steps_per_minute = 600
    log_path = np.cumsum(rng.normal(0, sigma_minute / math.sqrt(steps_per_minute), 600_000))
    prices = 100 * np.exp(log_path).reshape(-1, steps_per_minute)
    rows = [(p[0], p.max(), p.min(), p[-1]) for p in prices]
    assert m.sigma_per_bar(bars_from(rows)) == pytest.approx(sigma_minute, rel=0.05)


def test_expected_move_scales_with_sqrt_time():
    bars = bars_from([(100, 101, 99, 100)] * 16)
    assert m.expected_move(bars) == pytest.approx(m.sigma_per_bar(bars) * 4)


def test_efficiency_ratio_trend_vs_chop():
    trend = bars_from([(100 + i, 101 + i, 100 + i, 101 + i) for i in range(10)])
    chop = bars_from([(100, 101, 99, 101), (101, 101, 99, 100)] * 5)
    assert m.efficiency_ratio(trend) == pytest.approx(1.0)
    assert m.efficiency_ratio(chop) == pytest.approx(0.0)


def test_price_change():
    assert m.price_change(bars_from([(100, 101, 99, 100), (100, 106, 99, 105)])) == pytest.approx(
        0.05
    )


def test_resample_aligns_and_drops_incomplete_groups():
    # starts at minute 3 of a 5m bucket: first bucket (2 bars) is incomplete and dropped
    rows = [(i, i + 0.5, i - 0.5, i + 0.1, 1.0) for i in range(3, 15)]
    out = m.resample(bars_from(rows, start=T0 + 3 * MINUTE_MS), 5)
    assert list(out.ts) == [T0 + 5 * MINUTE_MS, T0 + 10 * MINUTE_MS]
    assert list(out.open) == [5, 10]
    assert list(out.high) == [9.5, 14.5]
    assert list(out.low) == [4.5, 9.5]
    assert list(out.close) == pytest.approx([9.1, 14.1])
    assert list(out.volume) == [5.0, 5.0]


def test_atr_constant_range():
    bars = bars_from([(100, 101, 99, 100)] * 20)
    assert m.atr(bars, period=14) == pytest.approx(0.02)
    assert math.isnan(m.atr(bars.tail(10), period=14))


def test_hourly_baseline_ratios_use_each_bars_hour():
    variance = np.full(24, 1e-6)
    variance[1] = 4e-6
    quote_volume = np.full(24, 100.0)
    baseline = m.HourlyBaseline(variance=variance, quote_volume=quote_volume)

    # One bar at 00:59 and one at 01:00, each with volume 1 @ price 100 → 100 quote each
    bars = bars_from([(100, 100.1, 99.9, 100)] * 2, start=T0 + 59 * MINUTE_MS)
    actual = float(np.mean(m.gk_variance(bars)))
    assert baseline.vol_ratio(bars) == pytest.approx(math.sqrt(actual / 2.5e-6))
    assert baseline.volume_ratio(bars) == pytest.approx(1.0)


def test_load_baselines_median_across_days(tmp_path):
    def day_hour(day: int, hour: int, spread: float, volume: float) -> list[Candle]:
        start = T0 + day * 86_400_000 + hour * 3_600_000
        return [
            Candle("binance", "X", start + i * MINUTE_MS, 100, 100 + spread, 100, 100, volume)
            for i in range(60)
        ]

    candles = []
    for day in range(7):
        # hour 0: calm except one crazy day; hour 5: only 3 days of data (falls back)
        candles += day_hour(day, 0, spread=1.0 if day != 3 else 20.0, volume=1.0)
    for day in range(3):
        candles += day_hour(day, 5, spread=5.0, volume=9.0)
    candles += day_hour(0, 7, spread=1.0, volume=1.0)[:30]  # too few minutes: ignored
    write_candles(tmp_path, candles)

    baseline = load_baselines(tmp_path, "binance", since_ms=0)["X"]
    calm = 0.5 * math.log(101 / 100) ** 2
    assert baseline.variance[0] == pytest.approx(calm)  # the crazy day does not move the median
    assert baseline.quote_volume[0] == pytest.approx(100.0)
    assert baseline.variance[5] == pytest.approx(calm)  # thin slot borrows symbol median
    assert baseline.variance[7] == pytest.approx(calm)  # missing slot too
    assert load_baselines(tmp_path, "bybit", since_ms=0) == {}
