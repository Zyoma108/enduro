"""Open interest change next to the price change over the same windows.

The level of open interest alone says little; its change against price says who is
acting: price up with OI up — new longs; price up with OI down — shorts closing; price
down with OI up — new shorts; price down with OI down — longs leaving. OI is compared in
coins (not USD), so a price move alone does not show up as an OI change.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from enduro.core.models import MINUTE_MS, Candle, OpenInterest

OI_WINDOWS_MIN: dict[str, int] = {"15m": 15, "1h": 60, "4h": 240}
# A window is reported only if the point found for its start is within this of it.
START_TOLERANCE_MIN = 5


def price_at(candles: Sequence[Candle], ts: int) -> float | None:
    """Close of the last 1m candle that closed at or before `ts`."""
    closed = [c for c in candles if c.ts + MINUTE_MS <= ts]
    return closed[-1].close if closed else None


def open_interest_view(
    points: Sequence[OpenInterest], candles: Sequence[Candle], now_ms: int
) -> dict[str, Any] | None:
    """Latest OI (coins and USD), its age, and per window the OI and price change (%)
    between the window's start and the latest point. `candles` are 1m, oldest first."""
    if not points:
        return None
    last = points[-1]
    price = price_at(candles, last.ts) or (candles[-1].close if candles else None)
    out: dict[str, Any] = {
        "coins": round(last.amount),
        "usd": None if price is None else round(last.amount * price),
        "as_of_min_ago": round((now_ms - last.ts) / MINUTE_MS, 1),
    }
    for name, minutes in OI_WINDOWS_MIN.items():
        target = last.ts - minutes * MINUTE_MS
        start = min(points, key=lambda p: abs(p.ts - target))
        if abs(start.ts - target) > START_TOLERANCE_MIN * MINUTE_MS or start is last:
            continue
        p0, p1 = price_at(candles, start.ts), price_at(candles, last.ts)
        out[name] = {
            "oi_pct": round((last.amount / start.amount - 1) * 100, 2) if start.amount else None,
            "price_pct": round((p1 / p0 - 1) * 100, 2) if p0 and p1 else None,
        }
    return out
