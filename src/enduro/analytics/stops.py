"""Where stops get swept: data for placing a stop beyond ordinary noise.

Intraday price routinely pokes past obvious swing highs/lows — where many stops sit —
and snaps back (seen live: STX wicked 0.7% through a local high in one 1m candle and
reversed below the entry). These functions measure, on recent 1m candles:
  - typical candle wicks (how far a single minute stretches beyond its body);
  - false breaks of swing levels: how far price went beyond a swing high/low before
    closing back inside — the distance a stop must clear to survive a sweep;
  - the nearest swing levels on the stop side of a long or a short.
They describe the market; where to put the stop is the agent's call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from enduro.analytics.metrics import Bars

SWING_K = 3  # a swing high is the highest high of 3 bars on each side
CONFIRM_BARS = 3  # a break counts as false if price closes back inside within this many bars


def _pct(values: list[float] | np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None


def wick_stats(bars: Bars) -> dict[str, float | None]:
    """Upper and lower 1m wicks as fractions of price: median, 90th percentile, max."""
    body_top = np.maximum(bars.open, bars.close)
    body_bottom = np.minimum(bars.open, bars.close)
    upper = (bars.high - body_top) / bars.close
    lower = (body_bottom - bars.low) / bars.close
    wicks = np.concatenate([upper, lower])
    return {
        "p50": _pct(wicks, 50),
        "p90": _pct(wicks, 90),
        "max": float(wicks.max()) if len(wicks) else None,
    }


@dataclass(frozen=True, slots=True)
class Swing:
    index: int
    ts: int
    price: float


def swing_levels(bars: Bars, k: int = SWING_K) -> tuple[list[Swing], list[Swing]]:
    """Swing highs and lows confirmed by `k` bars on each side.

    Strictly beyond the `k` bars before and at least as extreme as the `k` bars after,
    so a flat stretch (equal highs) yields no swing, or one at its first bar."""
    highs, lows = [], []
    for i in range(k, len(bars) - k):
        left, right = slice(i - k, i), slice(i + 1, i + k + 1)
        if bars.high[i] > bars.high[left].max() and bars.high[i] >= bars.high[right].max():
            highs.append(Swing(i, int(bars.ts[i]), float(bars.high[i])))
        if bars.low[i] < bars.low[left].min() and bars.low[i] <= bars.low[right].min():
            lows.append(Swing(i, int(bars.ts[i]), float(bars.low[i])))
    return highs, lows


def false_break_overshoots(
    bars: Bars, swings: list[Swing], above: bool, confirm: int = CONFIRM_BARS
) -> list[float]:
    """For each swing level, the first later break beyond it: if price closed back inside
    within `confirm` bars, record how far it went past the level (fraction of the level).
    `above=True` for swing highs (breaks upward), False for swing lows."""
    overshoots = []
    n = len(bars)
    for s in swings:
        start = s.index + SWING_K + 1
        for j in range(start, n):
            broke = bars.high[j] > s.price if above else bars.low[j] < s.price
            if not broke:
                continue
            end = min(n, j + confirm)
            back_inside = any(
                (bars.close[m] < s.price) if above else (bars.close[m] > s.price)
                for m in range(j, end)
            )
            if back_inside:
                extreme = bars.high[j:end].max() if above else bars.low[j:end].min()
                overshoots.append(abs(extreme - s.price) / s.price)
            break  # only the first break of each level
    return overshoots


def stop_context(bars: Bars, price: float, side: str, atr_5m: float | None) -> dict:
    """Everything the agent needs to place a stop for a `side` position at `price`."""
    highs, lows = swing_levels(bars)
    stop_above = side == "short"  # a short's stop is above price, a long's below
    swings = highs if stop_above else lows
    overshoots = false_break_overshoots(bars, swings, above=stop_above)
    typical_sweep = _pct(overshoots, 90)

    def in_atr(fraction: float) -> float | None:
        return round(fraction / atr_5m, 2) if atr_5m else None

    candidates = sorted(
        (s for s in swings if (s.price > price if stop_above else s.price < price)),
        key=lambda s: abs(s.price - price),
    )
    levels = []
    for s in candidates[:3]:
        distance = abs(s.price - price) / price
        level = {
            "price": s.price,
            "minutes_ago": len(bars) - 1 - s.index,
            "distance_pct": round(distance * 100, 3),
            "distance_atr_5m": in_atr(distance),
        }
        if typical_sweep is not None:
            beyond = s.price * (1 + typical_sweep if stop_above else 1 - typical_sweep)
            level["beyond_p90_sweep"] = float(f"{beyond:.8g}")
            level["beyond_p90_sweep_distance_pct"] = round(abs(beyond - price) / price * 100, 3)
        levels.append(level)

    def pct(x: float | None) -> float | None:
        return None if x is None else round(x * 100, 3)

    wicks = wick_stats(bars)
    return {
        "side": side,
        "price": price,
        "bars_1m": len(bars),
        "atr_5m_pct": pct(atr_5m),
        "wick_1m_pct": {k: pct(v) for k, v in wicks.items()},
        "false_breaks": {
            "levels": "swing highs" if stop_above else "swing lows",
            "count": len(overshoots),
            "overshoot_p50_pct": pct(_pct(overshoots, 50)),
            "overshoot_p90_pct": pct(typical_sweep),
            "overshoot_max_pct": pct(max(overshoots) if overshoots else None),
        },
        "nearest_levels_on_stop_side": levels,
    }
