from enduro.analytics.open_interest import open_interest_view, price_at
from enduro.core.models import MINUTE_MS, Candle, OpenInterest

T0 = 1_790_726_400_000  # minute boundary


def oi(minute, amount):
    return OpenInterest("binance", "X", T0 + minute * MINUTE_MS, amount)


def candle(minute, close):
    return Candle("binance", "X", T0 + minute * MINUTE_MS, close, close, close, close, 1.0)


def test_windows_pair_oi_change_with_price_change():
    points = [oi(m, 1000 + m) for m in range(0, 245, 5)]  # every 5 min for 4 hours
    candles = [candle(m, 100 + m * 0.1) for m in range(0, 245)]
    now = T0 + 242 * MINUTE_MS
    view = open_interest_view(points, candles, now)
    assert view["coins"] == 1240 and view["as_of_min_ago"] == 2.0
    assert view["usd"] == round(1240 * (100 + 239 * 0.1))  # close of the candle before 240
    assert view["15m"]["oi_pct"] == round((1240 / 1225 - 1) * 100, 2)
    assert view["15m"]["price_pct"] == round(((100 + 23.9) / (100 + 22.4) - 1) * 100, 2)
    assert view["4h"]["oi_pct"] == round((1240 / 1000 - 1) * 100, 2)
    assert view["4h"]["price_pct"] is None  # no candle closed before the first point


def test_windows_without_history_that_far_back_are_left_out():
    points = [oi(0, 500), oi(5, 510), oi(10, 520), oi(15, 530), oi(18, 600)]  # live point
    view = open_interest_view(points, [], T0 + 18 * MINUTE_MS)
    assert view["15m"]["oi_pct"] == 17.65  # 510 (minute 5, nearest to 18 - 15) -> 600
    assert "1h" not in view and "4h" not in view
    assert view["usd"] is None
    assert open_interest_view([], [], T0) is None


def test_price_at_uses_the_last_closed_candle():
    candles = [candle(0, 10), candle(1, 11), candle(2, 12)]
    assert price_at(candles, T0 + 2 * MINUTE_MS) == 11
    assert price_at(candles, T0 + 30_000) is None
