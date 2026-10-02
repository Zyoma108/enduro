"""Focus: real-time microstructure view of the coin(s) the agent is working.

Built from the live trade and order book streams of both exchanges:
  * flow  — who is pushing (taker delta), how hard (intensity), big prints, VWAP;
  * book  — cost of trading on the execution exchange: spread, depth, slippage;
  * cross — does the reference exchange (source of truth) confirm the move?

Windows are short (1m/5m/15m). Trade windows use exchange time `ts`; book samples and
cross-exchange comparisons use local `recv_ts` (exchange clocks are skewed).
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from enduro.core.models import MarketEvent, OrderBook, PriceLevel, Trade

WINDOWS_S: dict[str, int] = {"1m": 60, "5m": 300, "15m": 900}
HISTORY_MS = max(WINDOWS_S.values()) * 1000
BOOK_SAMPLE_MS = 1_000
LARGE_TRADE_QUANTILE = 0.99
MIN_TRADES_FOR_LARGE = 100
LARGE_TRADE_MIN_MEDIAN_MULTIPLE = 10
DEPTH_BANDS_BPS = (10, 25)
# Below these, a price move / taker imbalance is noise, not a direction to confirm.
CONFIRM_MIN_MOVE = 0.0002  # 2 bps
CONFIRM_MIN_DELTA = 0.1
DEFAULT_SLIPPAGE_NOTIONALS = (1_000.0, 10_000.0, 50_000.0)


# ---------------------------------------------------------------- pure computations


@dataclass(frozen=True, slots=True)
class FlowStats:
    seconds: float  # covered length: shorter than nominal while we have watched less
    trades: int
    notional: float  # USDT
    delta_ratio: float  # (taker buy - taker sell) / total notional, -1..1
    intensity: float  # trades per second
    intensity_vs_15m: float  # this window's trade rate vs the 15m average
    price_change: float  # fraction, first → last trade in the window
    vwap: float | None
    price_vs_vwap_bps: float | None
    large_buy_notional: float
    large_sell_notional: float


def flow_stats(
    trades: Sequence[Trade],  # window is (start_ms, end_ms]
    start_ms: int,
    end_ms: int,
    large_threshold: float | None,
    rate_15m: float,
) -> FlowStats:
    window = [t for t in trades if start_ms < t.ts <= end_ms]
    seconds = (end_ms - start_ms) / 1000
    notional = sum(t.notional for t in window)
    buys = sum(t.notional for t in window if t.side == "buy")
    sells = sum(t.notional for t in window if t.side == "sell")
    volume = sum(t.amount for t in window)
    vwap = notional / volume if volume else None
    last = window[-1].price if window else None
    large = [t for t in window if large_threshold is not None and t.notional >= large_threshold]
    intensity = len(window) / seconds if seconds else 0.0
    return FlowStats(
        seconds=seconds,
        trades=len(window),
        notional=notional,
        delta_ratio=(buys - sells) / (buys + sells) if buys + sells else 0.0,
        intensity=intensity,
        intensity_vs_15m=intensity / rate_15m if rate_15m else math.nan,
        price_change=window[-1].price / window[0].price - 1 if window and window[0].price else 0.0,
        vwap=vwap,
        price_vs_vwap_bps=(last / vwap - 1) * 1e4 if vwap and last else None,
        large_buy_notional=sum(t.notional for t in large if t.side == "buy"),
        large_sell_notional=sum(t.notional for t in large if t.side == "sell"),
    )


def large_trade_threshold(trades: Sequence[Trade]) -> float | None:
    """Notional above which a trade is a "large print": in the top 1% *and* at least 10x
    the median trade. The second condition matters when most trades have the same size
    (bots trading fixed lots): then the 99th percentile is just the typical size.
    None until there are enough trades to tell."""
    if len(trades) < MIN_TRADES_FOR_LARGE:
        return None
    notionals = np.array([t.notional for t in trades])
    return float(
        max(
            np.quantile(notionals, LARGE_TRADE_QUANTILE),
            LARGE_TRADE_MIN_MEDIAN_MULTIPLE * np.median(notionals),
        )
    )


def depth_within(levels: Sequence[PriceLevel], mid: float, bps: float) -> float:
    """Quote notional resting within `bps` of mid on one side of the book."""
    limit = bps / 1e4 * mid
    return sum(p * a for p, a in levels if abs(p - mid) <= limit)


def slippage_bps(levels: Sequence[PriceLevel], mid: float, notional: float) -> float | None:
    """Average fill price vs mid, in bps, for a market order of `notional` USDT walking
    `levels` (asks for a buy, bids for a sell). None if the visible book is too thin."""
    remaining, cost, qty = notional, 0.0, 0.0
    for price, amount in levels:
        take = min(amount, remaining / price)
        cost += take * price
        qty += take
        remaining -= take * price
        if remaining <= 1e-9:
            return abs(cost / qty / mid - 1) * 1e4
    return None


@dataclass(frozen=True, slots=True)
class BookStats:
    age_s: float  # since this snapshot was received; large = the stream is stale
    mid: float
    spread_bps: float
    spread_vs_15m: float  # current spread vs its 15m median
    visible_bps: tuple[float, float]  # how far from mid the received book reaches (bid, ask)
    # band bps -> (bid USDT, ask USDT); None for a side the visible book does not cover
    depth: dict[int, tuple[float | None, float | None]]
    slippage: dict[float, tuple[float | None, float | None]]  # notional -> (buy, sell) bps

    def imbalance(self, band: int) -> float | None:
        bid, ask = self.depth[band]
        if bid is None or ask is None or bid + ask == 0:
            return None
        return (bid - ask) / (bid + ask)


def book_stats(
    book: OrderBook,
    spread_history: Sequence[float],
    notionals: Sequence[float],
    now_ms: int | None = None,
) -> BookStats | None:
    mid, spread = book.mid, book.spread_bps
    if mid is None or spread is None:
        return None
    median = statistics.median(spread_history) if spread_history else spread
    visible = (
        (mid - book.bids[-1][0]) / mid * 1e4,
        (book.asks[-1][0] - mid) / mid * 1e4,
    )

    def covered(levels: Sequence[PriceLevel], reach: float, band: int) -> float | None:
        return depth_within(levels, mid, band) if reach >= band - 1e-6 else None

    return BookStats(
        age_s=max(0.0, ((now_ms if now_ms is not None else book.recv_ts) - book.recv_ts) / 1000),
        mid=mid,
        spread_bps=spread,
        spread_vs_15m=spread / median if median else math.nan,
        visible_bps=visible,
        depth={
            band: (covered(book.bids, visible[0], band), covered(book.asks, visible[1], band))
            for band in DEPTH_BANDS_BPS
        },
        slippage={
            n: (slippage_bps(book.asks, mid, n), slippage_bps(book.bids, mid, n)) for n in notionals
        },
    )


@dataclass(frozen=True, slots=True)
class CrossStats:
    basis_bps: float | None  # execution mid vs reference mid, now
    basis_mean_bps: float | None  # over the last 15m
    basis_std_bps: float | None
    reference_volume_share: dict[str, float | None]  # window -> reference / (both)
    confirms: dict[str, bool | None]  # window -> same direction of price and of pressure


def basis_series(
    reference: Sequence[tuple[int, float]], execution: Sequence[tuple[int, float]]
) -> list[float]:
    """Basis in bps for every second where both exchanges have a mid sample."""
    ref = {ts // BOOK_SAMPLE_MS: mid for ts, mid in reference}
    return [
        (mid / ref[sec] - 1) * 1e4 for ts, mid in execution if (sec := ts // BOOK_SAMPLE_MS) in ref
    ]


def confirms(reference: FlowStats, execution: FlowStats) -> bool | None:
    """Both exchanges move the same way and takers push the same way on both.

    None when there is nothing to compare (no trades, or flat price / balanced flow).
    """

    def direction(x: float, threshold: float) -> int:
        return 0 if abs(x) < threshold else int(np.sign(x))

    moves = [direction(f.price_change, CONFIRM_MIN_MOVE) for f in (reference, execution)]
    pushes = [direction(f.delta_ratio, CONFIRM_MIN_DELTA) for f in (reference, execution)]
    if not reference.trades or not execution.trades or 0 in moves + pushes:
        return None
    return moves[0] == moves[1] and pushes[0] == pushes[1]


# ---------------------------------------------------------------- live tracker


@dataclass(slots=True)
class _Stream:
    trades: deque[Trade] = field(default_factory=deque)
    book: OrderBook | None = None
    mids: deque[tuple[int, float]] = field(default_factory=deque)  # sampled (recv_ts, mid)
    spreads: deque[tuple[int, float]] = field(default_factory=deque)
    first_seen: int | None = None


@dataclass(frozen=True, slots=True)
class FocusSnapshot:
    symbol: str
    ts: int
    observed_s: float  # how long we have been watching: windows longer than this are partial
    flow: dict[str, dict[str, FlowStats]]  # exchange -> window -> stats
    book: dict[str, BookStats]  # exchange -> stats
    cross: CrossStats
    reference: str
    execution: str

    def to_summary(self) -> dict[str, Any]:
        """Compact form for the agent: numbers rounded, units in the names."""

        def r(x: float | None, digits: int = 2) -> float | None:
            return (
                None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, digits)
            )

        def flow(name: str, f: FlowStats) -> dict[str, Any]:
            partial = f.seconds < WINDOWS_S[name] - 1
            return {
                **({"partial_window_s": round(f.seconds)} if partial else {}),
                "change_pct": r(f.price_change * 100, 3),
                "notional_usd": round(f.notional),
                "delta_ratio": r(f.delta_ratio),
                "trades_per_s": r(f.intensity, 1),
                "intensity_vs_15m": r(f.intensity_vs_15m),
                "price_vs_vwap_bps": r(f.price_vs_vwap_bps, 1),
                "large_buy_usd": round(f.large_buy_notional),
                "large_sell_usd": round(f.large_sell_notional),
            }

        def book(b: BookStats) -> dict[str, Any]:
            return {
                "book_age_s": round(b.age_s, 1),
                "mid": float(f"{b.mid:.8g}"),
                "spread_bps": r(b.spread_bps),
                "spread_vs_15m": r(b.spread_vs_15m),
                "visible_book_bps": {"bid": r(b.visible_bps[0], 1), "ask": r(b.visible_bps[1], 1)},
                **{
                    f"depth_{band}bps_usd": {
                        "bid": None if bid is None else round(bid),
                        "ask": None if ask is None else round(ask),
                    }
                    for band, (bid, ask) in b.depth.items()
                },
                f"imbalance_{DEPTH_BANDS_BPS[0]}bps": r(b.imbalance(DEPTH_BANDS_BPS[0])),
                "slippage_bps": {
                    f"{int(n)}usd": {"buy": r(buy), "sell": r(sell)}
                    for n, (buy, sell) in b.slippage.items()
                },
            }

        return {
            "symbol": self.symbol,
            "observed_s": round(self.observed_s),
            "flow": {ex: _distinct_windows(by_w, flow) for ex, by_w in self.flow.items()},
            "book": {ex: book(b) for ex, b in self.book.items()},
            "cross": {
                "basis_bps": r(self.cross.basis_bps),
                "basis_mean_15m_bps": r(self.cross.basis_mean_bps),
                "basis_std_15m_bps": r(self.cross.basis_std_bps),
                f"{self.reference}_volume_share": {
                    w: r(v) for w, v in self.cross.reference_volume_share.items()
                },
                f"{self.reference}_confirms": self.cross.confirms,
            },
        }


def _distinct_windows(
    by_window: dict[str, FlowStats], render: Callable[[str, FlowStats], dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Render windows, skipping longer ones that cover no more time than a shorter one
    (right after focusing, 5m and 15m would just repeat 1m and waste the agent's context)."""
    out: dict[str, dict[str, Any]] = {}
    covered = 0.0
    for name, stats in by_window.items():
        if out and stats.seconds <= covered + 1:
            continue
        out[name] = render(name, stats)
        covered = stats.seconds
    return out


class FocusTracker:
    def __init__(
        self,
        reference: str,
        execution: str,
        slippage_notionals: Sequence[float] = DEFAULT_SLIPPAGE_NOTIONALS,
    ) -> None:
        self.reference = reference
        self.execution = execution
        self.slippage_notionals = tuple(slippage_notionals)
        self._streams: dict[tuple[str, str], _Stream] = {}

    def on_event(self, event: MarketEvent) -> None:
        stream = self._streams.setdefault((event.exchange, event.symbol), _Stream())
        if stream.first_seen is None:
            stream.first_seen = event.recv_ts
        cutoff = event.recv_ts - HISTORY_MS
        if isinstance(event, Trade):
            if not (event.price > 0 and event.amount > 0):
                return  # malformed print (seen live: a zero-price 1000PEPE trade)
            stream.trades.append(event)
            while stream.trades and stream.trades[0].ts < cutoff:
                stream.trades.popleft()
            return
        stream.book = event
        if event.mid is None or event.spread_bps is None:
            return
        if not stream.mids or event.recv_ts - stream.mids[-1][0] >= BOOK_SAMPLE_MS:
            stream.mids.append((event.recv_ts, event.mid))
            stream.spreads.append((event.recv_ts, event.spread_bps))
            for samples in (stream.mids, stream.spreads):
                while samples and samples[0][0] < cutoff:
                    samples.popleft()

    def latest_book(self, exchange: str, symbol: str) -> OrderBook | None:
        stream = self._streams.get((exchange, symbol))
        return stream.book if stream else None

    def drop(self, symbol: str) -> None:
        """Forget a symbol that left the focus."""
        for key in [k for k in self._streams if k[1] == symbol]:
            del self._streams[key]

    def snapshot(self, symbol: str, now_ms: int) -> FocusSnapshot | None:
        streams = {ex: self._streams.get((ex, symbol)) for ex in (self.reference, self.execution)}
        if not any(streams.values()):
            return None
        first_seen = min(s.first_seen for s in streams.values() if s and s.first_seen)

        flow: dict[str, dict[str, FlowStats]] = {}
        books: dict[str, BookStats] = {}
        for ex, stream in streams.items():
            if stream is None:
                continue
            trades = list(stream.trades)
            span_15m = min(HISTORY_MS, max(now_ms - first_seen, 1)) / 1000
            recent = [t for t in trades if t.ts >= now_ms - HISTORY_MS]
            rate_15m = len(recent) / span_15m
            threshold = large_trade_threshold(recent)
            # A window cannot start before we started watching, or rates would be diluted.
            flow[ex] = {
                name: flow_stats(
                    trades, max(now_ms - sec * 1000, first_seen - 1), now_ms, threshold, rate_15m
                )
                for name, sec in WINDOWS_S.items()
            }
            if stream.book is not None:
                stats = book_stats(
                    stream.book, [s for _, s in stream.spreads], self.slippage_notionals, now_ms
                )
                if stats is not None:
                    books[ex] = stats

        return FocusSnapshot(
            symbol=symbol,
            ts=now_ms,
            observed_s=(now_ms - first_seen) / 1000,
            flow=flow,
            book=books,
            cross=self._cross(streams, flow, books),
            reference=self.reference,
            execution=self.execution,
        )

    def _cross(
        self,
        streams: dict[str, _Stream | None],
        flow: dict[str, dict[str, FlowStats]],
        books: dict[str, BookStats],
    ) -> CrossStats:
        ref, exe = self.reference, self.execution
        basis = None
        if ref in books and exe in books:
            basis = (books[exe].mid / books[ref].mid - 1) * 1e4
        series: list[float] = []
        if streams[ref] and streams[exe]:
            series = basis_series(streams[ref].mids, streams[exe].mids)
        share: dict[str, float | None] = {}
        agree: dict[str, bool | None] = {}
        for name in WINDOWS_S:
            r, e = flow.get(ref, {}).get(name), flow.get(exe, {}).get(name)
            total = (r.notional if r else 0.0) + (e.notional if e else 0.0)
            share[name] = (r.notional if r else 0.0) / total if total else None
            agree[name] = confirms(r, e) if r and e else None
        return CrossStats(
            basis_bps=basis,
            basis_mean_bps=statistics.fmean(series) if series else None,
            basis_std_bps=statistics.pstdev(series) if len(series) > 1 else None,
            reference_volume_share=share,
            confirms=agree,
        )
