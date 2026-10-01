import pytest

from enduro.analytics import stops
from enduro.analytics.metrics import Bars
from enduro.core.models import MINUTE_MS, Candle

T0 = 1_790_726_400_000


def bars(rows) -> Bars:
    """rows: (open, high, low, close) per minute."""
    return Bars.from_candles(
        [
            Candle("x", "X", T0 + i * MINUTE_MS, o, h, lo, c, 1.0)
            for i, (o, h, lo, c) in enumerate(rows)
        ]
    )


def flat(n, price=100.0):
    return [(price, price + 0.1, price - 0.1, price)] * n


def test_wick_stats():
    b = bars([(100, 101, 99.5, 100.5), (100, 100, 100, 100)])
    w = stops.wick_stats(b)
    # wicks: upper 0.5/100.5, lower 0.5/100.5, then 0 and 0
    assert w["max"] == pytest.approx(0.5 / 100.5)
    assert w["p50"] == pytest.approx(0.25 / 100.5, rel=1e-6)


def test_swing_levels():
    rows = flat(3) + [(100, 103, 99.9, 100)] + flat(3) + [(100, 100.1, 97, 100)] + flat(3)
    highs, lows = stops.swing_levels(bars(rows))
    assert [h.price for h in highs] == [103]
    assert [lo.price for lo in lows] == [97]
    assert highs[0].index == 3 and lows[0].index == 7


def test_false_break_overshoot_of_a_swing_high():
    rows = (
        flat(3)
        + [(100, 102, 99.9, 100)]  # swing high 102 at index 3
        + flat(3)
        + [(100, 102.6, 100, 101)]  # pokes 0.6 above, closes back below 102: false break
        + flat(2)
    )
    b = bars(rows)
    highs, _ = stops.swing_levels(b)
    assert stops.false_break_overshoots(b, highs, above=True) == [pytest.approx(0.6 / 102)]


def test_real_breakout_is_not_a_false_break():
    rows = (
        flat(3)
        + [(100, 102, 99.9, 100)]
        + flat(3)
        + [(100, 103, 100, 102.8)]
        + [(103, 104, 102.5, 103.5)] * 3
    )
    b = bars(rows)
    highs, _ = stops.swing_levels(b)
    assert stops.false_break_overshoots(b, highs, above=True) == []


def test_stop_context_for_a_short_lists_levels_above_with_sweep_margin():
    rows = (
        flat(3)
        + [(100, 102, 99.9, 100)]  # swing high 102
        + flat(3)
        + [(100, 102.6, 100, 101)]  # false break: +0.588%
        + flat(3)
        + [(100, 101, 99.9, 100)]  # swing high 101 (nearer)
        + flat(4)
    )
    ctx = stops.stop_context(bars(rows), price=100.0, side="short", atr_5m=0.01)
    assert ctx["false_breaks"]["count"] == 1
    sweep = 0.6 / 102
    assert ctx["false_breaks"]["overshoot_p90_pct"] == pytest.approx(sweep * 100, abs=1e-3)
    nearest = ctx["nearest_levels_on_stop_side"]
    assert [lvl["price"] for lvl in nearest][:2] == [101, 102]
    assert nearest[0]["distance_pct"] == pytest.approx(1.0)
    assert nearest[0]["distance_atr_5m"] == pytest.approx(1.0)
    assert nearest[0]["beyond_p90_sweep"] == pytest.approx(101 * (1 + sweep))


def test_stop_context_for_a_long_uses_swing_lows_below():
    rows = flat(3) + [(100, 100.1, 98, 100)] + flat(5)
    ctx = stops.stop_context(bars(rows), price=100.0, side="long", atr_5m=None)
    assert ctx["false_breaks"]["levels"] == "swing lows"
    assert ctx["nearest_levels_on_stop_side"][0]["price"] == 98
    assert ctx["nearest_levels_on_stop_side"][0]["distance_atr_5m"] is None
    assert "beyond_p90_sweep" not in ctx["nearest_levels_on_stop_side"][0]  # no false breaks yet


def test_price_scale_moves_prices_to_the_execution_exchange_but_not_distances():
    rows = flat(3) + [(100, 102, 99.9, 100)] + flat(5)
    plain = stops.stop_context(bars(rows), price=100.0, side="short", atr_5m=0.01)
    shifted = stops.stop_context(
        bars(rows), price=100.0, side="short", atr_5m=0.01, price_scale=0.999
    )
    assert shifted["price"] == pytest.approx(99.9)
    level = shifted["nearest_levels_on_stop_side"][0]
    level0 = plain["nearest_levels_on_stop_side"][0]
    assert level["price"] == pytest.approx(102 * 0.999)
    assert level["distance_pct"] == level0["distance_pct"]
