"""Impulses and market context from the radar's 1m candles.

An impulse is a fresh break of a coin's recent range on unusually heavy volume: the last
5 minutes closed beyond the 2-hour high (or low) while trading many times the coin's
average volume of the last day. These are the pumps and dumps worth a look within
minutes — the agent is woken for them instead of finding them on its next scheduled check.

On Binance 1m history (2026-09-02 … 10-08, 87 liquid coins) breaks up on >10x volume
continued more often than not while the market rose, and breaks down were bought back
more often than they continued; see docs/experiments.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from enduro.core.models import Candle

IMPULSE_WINDOW_MIN = 5  # the move being judged
RANGE_MIN = 120  # the range it breaks
DAY_MIN = 24 * 60
MARKET_WINDOWS_MIN = {"15m": 15, "1h": 60, "4h": 240, "24h": DAY_MIN}


@dataclass(frozen=True, slots=True)
class Impulse:
    symbol: str
    direction: str  # "up" | "down"
    ts: int  # open time of the last candle of the impulse
    change_pct: float  # over the impulse window
    volume_vs_24h: float  # volume per minute in the window vs the day's average
    beyond_range_pct: float  # how far the close is past the 2h high / low

    def describe(self) -> str:
        level = "high" if self.direction == "up" else "low"
        return (
            f"impulse {self.direction} on {self.symbol}: {self.change_pct:+.2f}% in "
            f"{IMPULSE_WINDOW_MIN} min on {self.volume_vs_24h:.0f}x its average volume, "
            f"closed {self.beyond_range_pct:.2f}% past its {RANGE_MIN // 60}h {level}"
        )


def detect_impulse(
    symbol: str, candles: Sequence[Candle], min_volume_vs_24h: float
) -> Impulse | None:
    """An impulse ending at the last closed candle, if there is one."""
    if len(candles) < RANGE_MIN + IMPULSE_WINDOW_MIN:
        return None
    window = candles[-IMPULSE_WINDOW_MIN:]
    before = candles[-(RANGE_MIN + IMPULSE_WINDOW_MIN) : -IMPULSE_WINDOW_MIN]
    day = candles[-DAY_MIN:]
    day_qv = sum(c.close * c.volume for c in day) / len(day)
    if day_qv <= 0:
        return None
    volume_ratio = sum(c.close * c.volume for c in window) / len(window) / day_qv
    if volume_ratio < min_volume_vs_24h:
        return None
    close = window[-1].close
    high = max(c.high for c in before)
    low = min(c.low for c in before)
    if close > high:
        direction, beyond = "up", close / high - 1
    elif close < low:
        direction, beyond = "down", 1 - close / low
    else:
        return None
    return Impulse(
        symbol=symbol,
        direction=direction,
        ts=window[-1].ts,
        change_pct=round((close / window[0].open - 1) * 100, 2),
        volume_vs_24h=round(volume_ratio, 1),
        beyond_range_pct=round(beyond * 100, 2),
    )


def _change_pct(candles: Sequence[Candle], minutes: int) -> float | None:
    if len(candles) < minutes:
        return None
    return round((candles[-1].close / candles[-minutes].open - 1) * 100, 2)


def market_context(
    leaders: dict[str, Sequence[Candle]], universe: dict[str, Sequence[Candle]]
) -> dict[str, Any]:
    """Where the market goes: the leaders' price change over several windows, and breadth
    — the share of the universe up over each window and its median change."""
    out: dict[str, Any] = {}
    for symbol, candles in leaders.items():
        changes = {k: _change_pct(candles, m) for k, m in MARKET_WINDOWS_MIN.items()}
        out[symbol.split("/")[0]] = {k: v for k, v in changes.items() if v is not None}
    breadth: dict[str, Any] = {}
    for name, minutes in MARKET_WINDOWS_MIN.items():
        changes = sorted(
            c for cs in universe.values() if (c := _change_pct(cs, minutes)) is not None
        )
        if changes:
            breadth[name] = {
                "share_up": round(sum(1 for c in changes if c > 0) / len(changes), 2),
                "median_change_pct": changes[len(changes) // 2],
            }
    if breadth:
        out["universe"] = breadth
    return out
