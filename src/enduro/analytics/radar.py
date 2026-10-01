"""Market radar: ranks the universe by how unusually alive each symbol is right now.

Metrics are computed on the reference exchange (source of truth) from 1m candles and
compared against two references:
  * "usual" — the symbol's own hour-of-day norm over weeks: is this coin in play at all?
    ("3x" = three times what is normal for this coin at this time of day);
  * "24h" — the symbol's own average over the last day: is it heating up right now,
    or just as hot as it has been all day?
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from enduro.analytics import metrics as m
from enduro.core.models import Candle

WINDOWS: dict[str, int] = {"5m": 5, "15m": 15, "1h": 60}
SCORE_WINDOW = "15m"
DAY_MINUTES = 24 * 60
LOOKBACK_MINUTES = DAY_MINUTES


@dataclass(frozen=True, slots=True)
class WindowStats:
    change: float  # price change over the window, fraction
    expected_move: float  # 1-sigma move over the window at its own realized vol, fraction
    vol_ratio: float  # realized vol vs usual for these hours (1.0 = normal)
    volume_ratio: float  # RVOL: traded volume vs usual for these hours
    vol_vs_24h: float  # realized vol vs the last 24h average (1.0 = as usual today)
    volume_vs_24h: float  # volume per minute vs the last 24h average
    efficiency: float  # 1 = clean trend, 0 = chop


@dataclass(frozen=True, slots=True)
class RadarRow:
    symbol: str
    price: float
    ts: int  # open time of the last closed candle
    windows: dict[str, WindowStats]
    day_volume_ratio: float  # last 24h volume vs usual: >1 means the coin is "in play"
    expansion: float  # 5m vol / 1h vol: >1 accelerating, <1 calming down
    atr_5m: float  # ATR(14) on 5m bars, fraction of price
    move_vs_fees: float  # 15m expected move / round-trip taker fees
    score: float  # geometric mean of 15m vol_ratio and volume_ratio

    def to_summary(self) -> dict[str, Any]:
        """Compact, human-readable form for the agent and logs."""

        def pct(x: float) -> float | None:
            return None if math.isnan(x) else round(x * 100, 3)

        def num(x: float, digits: int = 2) -> float | None:
            return None if math.isnan(x) else round(x, digits)

        return {
            "symbol": self.symbol,
            "price": self.price,
            "score": num(self.score),
            "day_volume_vs_usual": num(self.day_volume_ratio),
            "expansion": num(self.expansion),
            "atr_5m_pct": pct(self.atr_5m),
            "move_vs_fees": num(self.move_vs_fees, 1),
            **{
                name: {
                    "change_pct": pct(w.change),
                    "expected_move_pct": pct(w.expected_move),
                    "vol_vs_usual": num(w.vol_ratio),
                    "volume_vs_usual": num(w.volume_ratio),
                    "vol_vs_24h": num(w.vol_vs_24h),
                    "volume_vs_24h": num(w.volume_vs_24h),
                    "efficiency": num(w.efficiency),
                }
                for name, w in self.windows.items()
            },
        }


def analyze(
    symbol: str, bars: m.Bars, baseline: m.HourlyBaseline, round_trip_fee: float
) -> RadarRow | None:
    """Radar row for one symbol, or None if there is not enough history yet."""
    if len(bars) < max(WINDOWS.values()):
        return None
    day = bars.tail(DAY_MINUTES)
    day_var = float(np.mean(m.gk_variance(day)))
    day_qv = float(np.mean(day.quote_volume))
    windows = {}
    for name, minutes in WINDOWS.items():
        w = bars.tail(minutes)
        windows[name] = WindowStats(
            change=m.price_change(w),
            expected_move=m.expected_move(w),
            vol_ratio=baseline.vol_ratio(w),
            volume_ratio=baseline.volume_ratio(w),
            vol_vs_24h=_ratio(math.sqrt(float(np.mean(m.gk_variance(w)))), math.sqrt(day_var)),
            volume_vs_24h=_ratio(float(np.mean(w.quote_volume)), day_qv),
            efficiency=m.efficiency_ratio(w),
        )
    short, long_ = m.sigma_per_bar(bars.tail(5)), m.sigma_per_bar(bars.tail(60))
    key = windows[SCORE_WINDOW]
    return RadarRow(
        symbol=symbol,
        price=float(bars.close[-1]),
        ts=int(bars.ts[-1]),
        windows=windows,
        day_volume_ratio=baseline.volume_ratio(day),
        expansion=short / long_ if long_ > 0 else math.nan,
        atr_5m=m.atr(m.resample(bars, 5)),
        move_vs_fees=key.expected_move / round_trip_fee if round_trip_fee > 0 else math.nan,
        score=math.sqrt(max(key.vol_ratio, 0) * max(key.volume_ratio, 0)),
    )


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else math.nan


class Radar:
    """Keeps a rolling window of 1m candles per symbol and ranks symbols on demand."""

    def __init__(
        self,
        baselines: dict[str, m.HourlyBaseline],
        round_trip_fee: float,
        lookback_minutes: int = LOOKBACK_MINUTES,
    ) -> None:
        self.baselines = baselines
        self.round_trip_fee = round_trip_fee
        self._candles: dict[str, deque[Candle]] = {}
        self._lookback = lookback_minutes

    def candles(self, symbol: str, minutes: int) -> list[Candle]:
        """The last `minutes` closed 1m candles held for `symbol` (oldest first)."""
        buf = self._candles.get(symbol)
        return list(buf)[-minutes:] if buf else []

    def last_ts(self) -> dict[str, int]:
        return {s: c[-1].ts for s, c in self._candles.items() if c}

    def add(self, candles: Iterable[Candle]) -> None:
        for c in candles:
            buf = self._candles.setdefault(c.symbol, deque(maxlen=self._lookback))
            if not buf or c.ts > buf[-1].ts:
                buf.append(c)

    def scan(self, symbols: Sequence[str] | None = None) -> list[RadarRow]:
        rows = []
        for symbol in symbols if symbols is not None else list(self._candles):
            baseline = self.baselines.get(symbol)
            candles = self._candles.get(symbol)
            if baseline is None or not candles:
                continue
            row = analyze(symbol, m.Bars.from_candles(candles), baseline, self.round_trip_fee)
            if row is not None:
                rows.append(row)
        return sorted(rows, key=lambda r: -r.score if not math.isnan(r.score) else math.inf)
