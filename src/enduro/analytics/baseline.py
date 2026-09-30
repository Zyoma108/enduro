"""Builds per-symbol hour-of-day baselines from stored candle history."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from enduro.analytics.metrics import HOUR_MS, HourlyBaseline
from enduro.storage import store

# A day's hour counts only if most of its minutes exist (listing gaps, outages).
MIN_MINUTES_PER_HOUR = 45
# An hour-of-day slot needs this many days of data, otherwise it borrows the symbol median.
MIN_DAYS_PER_SLOT = 5

_BASELINE_SQL = f"""
WITH hourly AS (
    SELECT symbol, ts // {HOUR_MS} AS hour_idx,
           avg(greatest(0, 0.5 * ln(high / low) ^ 2
                          - (2 * ln(2) - 1) * ln(close / open) ^ 2)) AS variance,
           avg(close * volume) AS quote_volume
    FROM candles
    WHERE exchange = ? AND timeframe = '1m' AND ts >= ? AND low > 0 AND open > 0
    GROUP BY ALL
    HAVING count(*) >= {MIN_MINUTES_PER_HOUR}
)
SELECT symbol, hour_idx % 24 AS hour, median(variance), median(quote_volume), count(*)
FROM hourly
GROUP BY ALL
"""


def load_baselines(root: Path | str, exchange: str, since_ms: int) -> dict[str, HourlyBaseline]:
    """Typical activity per UTC hour: median across days of each day's hourly mean."""
    con = store.connect(root)
    if not store.has_view(con, "candles"):
        return {}
    slots: dict[str, dict[int, tuple[float, float, int]]] = defaultdict(dict)
    for symbol, hour, variance, quote_volume, days in con.execute(
        _BASELINE_SQL, [exchange, since_ms]
    ).fetchall():
        slots[symbol][int(hour)] = (variance, quote_volume, days)
    return {symbol: _fill(by_hour) for symbol, by_hour in slots.items()}


def _fill(by_hour: dict[int, tuple[float, float, int]]) -> HourlyBaseline:
    """Use the symbol-wide median for hours with too little history."""
    good = {h: v for h, v in by_hour.items() if v[2] >= MIN_DAYS_PER_SLOT} or by_hour
    fallback_var = float(np.median([v[0] for v in good.values()]))
    fallback_qv = float(np.median([v[1] for v in good.values()]))
    variance = np.array([good[h][0] if h in good else fallback_var for h in range(24)])
    quote_volume = np.array([good[h][1] if h in good else fallback_qv for h in range(24)])
    return HourlyBaseline(variance=variance, quote_volume=quote_volume)
