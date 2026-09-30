"""Volatility, trend and activity metrics over 1m candles.

Pure functions over `Bars` (column arrays, oldest first). Internally everything is in
fractions; conversion to % / bps happens at the presentation layer.

Volatility uses the Garman-Klass estimator per bar, which uses open/high/low/close and
is several times more efficient than close-to-close returns on the same number of bars.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from enduro.core.models import MINUTE_MS, Candle

HOUR_MS = 60 * MINUTE_MS
_GK_CLOSE_COEF = 2 * math.log(2) - 1


@dataclass(frozen=True, slots=True)
class Bars:
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray  # base currency

    @classmethod
    def from_candles(cls, candles: Sequence[Candle]) -> Bars:
        return cls(
            ts=np.array([c.ts for c in candles], dtype=np.int64),
            open=np.array([c.open for c in candles], dtype=float),
            high=np.array([c.high for c in candles], dtype=float),
            low=np.array([c.low for c in candles], dtype=float),
            close=np.array([c.close for c in candles], dtype=float),
            volume=np.array([c.volume for c in candles], dtype=float),
        )

    def __len__(self) -> int:
        return len(self.ts)

    def tail(self, n: int) -> Bars:
        return Bars(*(a[-n:] for a in self._arrays()))

    @property
    def quote_volume(self) -> np.ndarray:
        return self.close * self.volume

    def _arrays(self) -> tuple[np.ndarray, ...]:
        return (self.ts, self.open, self.high, self.low, self.close, self.volume)


def gk_variance(bars: Bars) -> np.ndarray:
    """Garman-Klass variance of log price per bar (clipped at 0)."""
    hl = np.log(bars.high / bars.low)
    co = np.log(bars.close / bars.open)
    return np.maximum(0.0, 0.5 * hl**2 - _GK_CLOSE_COEF * co**2)


def sigma_per_bar(bars: Bars) -> float:
    """Realized volatility per bar (std of log return)."""
    return math.sqrt(float(np.mean(gk_variance(bars)))) if len(bars) else math.nan


def expected_move(bars: Bars) -> float:
    """1-sigma move over a span as long as `bars`, at the volatility realized within them."""
    return sigma_per_bar(bars) * math.sqrt(len(bars))


def price_change(bars: Bars) -> float:
    return float(bars.close[-1] / bars.open[0] - 1)


def efficiency_ratio(bars: Bars) -> float:
    """Kaufman efficiency: |net move| / path length. ~1 = clean trend, ~0 = chop."""
    prices = np.concatenate(([bars.open[0]], bars.close))
    path = float(np.sum(np.abs(np.diff(prices))))
    return abs(float(prices[-1] - prices[0])) / path if path > 0 else 0.0


def resample(bars: Bars, minutes: int) -> Bars:
    """Aggregate 1m bars into `minutes`-bars aligned to epoch; incomplete groups dropped."""
    span = minutes * MINUTE_MS
    group = bars.ts // span
    starts = np.flatnonzero(np.diff(group, prepend=group[0] - 1))
    ends = np.append(starts[1:], len(bars))
    full = (ends - starts) == minutes
    starts, ends = starts[full], ends[full]
    return Bars(
        ts=group[starts] * span,
        open=bars.open[starts],
        high=np.maximum.reduceat(bars.high, starts) if len(starts) else bars.high[:0],
        low=np.minimum.reduceat(bars.low, starts) if len(starts) else bars.low[:0],
        close=bars.close[ends - 1],
        volume=np.add.reduceat(bars.volume, starts) if len(starts) else bars.volume[:0],
    )


def atr(bars: Bars, period: int = 14) -> float:
    """Wilder's Average True Range as a fraction of the last close."""
    if len(bars) < period + 1:
        return math.nan
    prev_close = bars.close[:-1]
    tr = np.maximum.reduce(
        [
            bars.high[1:] - bars.low[1:],
            np.abs(bars.high[1:] - prev_close),
            np.abs(bars.low[1:] - prev_close),
        ]
    )
    value = float(np.mean(tr[:period]))
    for x in tr[period:]:
        value = (value * (period - 1) + x) / period
    return value / float(bars.close[-1])


@dataclass(frozen=True, slots=True)
class HourlyBaseline:
    """Typical per-minute GK variance and quote volume for each UTC hour of day."""

    variance: np.ndarray  # shape (24,)
    quote_volume: np.ndarray  # shape (24,)

    def _by_hour(self, values: np.ndarray, ts: np.ndarray) -> np.ndarray:
        return values[(ts // HOUR_MS) % 24]

    def vol_ratio(self, bars: Bars) -> float:
        """Realized volatility relative to what is usual for these hours (1.0 = normal)."""
        expected = float(np.mean(self._by_hour(self.variance, bars.ts)))
        actual = float(np.mean(gk_variance(bars)))
        return math.sqrt(actual / expected) if expected > 0 else math.nan

    def volume_ratio(self, bars: Bars) -> float:
        """Relative volume (RVOL): traded quote volume vs usual for these hours."""
        expected = float(np.sum(self._by_hour(self.quote_volume, bars.ts)))
        return float(np.sum(bars.quote_volume)) / expected if expected > 0 else math.nan
