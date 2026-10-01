import pytest

from enduro.agent.alerts import AlertBook
from enduro.core.models import MINUTE_MS, Candle

T0 = 1_790_726_400_000


def bar(minute: int, close: float) -> Candle:
    return Candle("binance", "X", T0 + minute * MINUTE_MS, close, close, close, close, 1.0)


def test_only_candles_closed_after_the_alert_count():
    book = AlertBook()
    # set during minute 5 (after the minute-4 candle closed, before minute 5 closes)
    book.add("X", 100.0, "above", "breakout", ttl_min=60, now_ms=T0 + 5 * MINUTE_MS + 10_000)
    bars = [bar(3, 101.0), bar(4, 102.0)]  # closed before the alert: ignored
    assert book.check(lambda s: bars, T0 + 6 * MINUTE_MS) == []
    bars.append(bar(5, 100.0))  # not strictly above
    assert book.check(lambda s: bars, T0 + 6 * MINUTE_MS) == []
    bars.append(bar(6, 100.5))
    [fired] = book.check(lambda s: bars, T0 + 7 * MINUTE_MS)
    assert fired.candle.close == 100.5
    assert book.active(T0 + 7 * MINUTE_MS) == []  # fires once


def test_alerts_expire_and_are_limited():
    book = AlertBook(max_alerts=2)
    book.add("X", 1.0, "below", "a", ttl_min=1, now_ms=T0)
    book.add("X", 2.0, "below", "b", ttl_min=10, now_ms=T0)
    with pytest.raises(ValueError, match="at most 2"):
        book.add("X", 3.0, "below", "c", ttl_min=10, now_ms=T0)
    [expired] = book.prune(T0 + MINUTE_MS)
    assert expired.note == "a"
    book.add("X", 3.0, "below", "c", ttl_min=10, now_ms=T0 + MINUTE_MS)  # room again
    assert [a.id for a in book.active(T0 + MINUTE_MS)] == [2, 3]
    assert book.cancel(2).note == "b" and book.cancel(2) is None


@pytest.mark.parametrize(
    ("level", "direction", "ttl", "fragment"),
    [(1.0, "sideways", 10, "direction"), (0.0, "above", 10, "positive"), (1.0, "above", 0, "1..")],
)
def test_bad_alerts_are_rejected(level, direction, ttl, fragment):
    with pytest.raises(ValueError, match=fragment):
        AlertBook().add("X", level, direction, "n", ttl_min=ttl, now_ms=T0)
