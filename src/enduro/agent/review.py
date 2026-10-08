"""Hindsight on the agent's own closed trades, shown next to each exchange record.

The agent sees PnL alone otherwise, which says little about *why* a trade lost: was the
entry late, the stop too tight, or the exit premature? For each closed trade we add,
from the execution exchange's 1m candles:

  * how far price went for and against the position while it was open;
  * where price was 15 / 30 / 60 minutes after the exit, signed so that + means it kept
    going the trade's way (the exit was early) and - means the exit saved money;
  * for exits the agent made itself: whether the stop or the take profit it had in place
    would have been hit first within the hour — i.e. what holding would have done.

Candles are treated coarsely: within one bar we cannot tell whether the high or the low
came first, so a bar touching both levels counts as the stop (the cautious reading).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from enduro.core.models import MINUTE_MS, Candle
from enduro.execution.models import ClosedTrade

AFTER_EXIT_MIN = (15, 30, 60)
HOLD_CHECK_MIN = 60  # how long after an exit we look for the stop / take profit


@dataclass(frozen=True, slots=True)
class TradeContext:
    """What the journal knows about how a closed trade was opened and protected."""

    opened_ms: int | None = None
    thesis: str | None = None
    stop: float | None = None  # in place at the exit (after any update_protection)
    take: float | None = None


def trade_context(
    trade: ClosedTrade, orders: Sequence[dict[str, Any]], risks: Sequence[dict[str, Any]]
) -> TradeContext:
    """Find the open order behind `trade` (the latest real open of that symbol and side
    before the close), its thesis, and the stop / take profit in place at the exit."""

    def filled_ms(order: dict[str, Any]) -> int:
        return (order.get("result") or {}).get("ts") or order["ts"]

    opens = [
        o
        for o in orders
        if o.get("action") == "open"
        and not o.get("dry_run")
        and (o.get("request") or {}).get("symbol") == trade.symbol
        and o["request"].get("position_side") == trade.side
        and filled_ms(o) <= trade.closed_ms
    ]
    if not opens:
        return TradeContext()
    opened = opens[-1]
    opened_ms = filled_ms(opened)
    thesis = None
    for r in risks:
        intent = r.get("intent") or {}
        if (
            r["ts"] <= opened["ts"]
            and intent.get("symbol") == trade.symbol
            and intent.get("side") == trade.side
            and (r.get("decision") or {}).get("approved")
        ):
            thesis = r.get("thesis")
    stop = opened["request"].get("stop_loss")
    take = opened["request"].get("take_profit") or opened.get("take_profit")
    for r in risks:  # protection moved while the position was open
        position = r.get("position") or {}
        if (
            r.get("protection")
            and r.get("ok")
            and opened["ts"] <= r["ts"] <= trade.closed_ms
            and position.get("symbol") == trade.symbol
            and position.get("side") == trade.side
        ):
            stop = r["protection"].get("stop_loss") or stop
            take = r["protection"].get("take_profit") or take
    return TradeContext(opened_ms=opened_ms, thesis=thesis, stop=stop, take=take)


def _signed_pct(side: str, start: float, end: float) -> float:
    """Move from `start` to `end` in percent, positive in the trade's favour."""
    move = (end / start - 1) * 100
    return round(move if side == "long" else -move, 2)


def review_trade(
    *,
    side: str,
    entry: float,
    exit_price: float,
    opened_ms: int | None,
    closed_ms: int,
    stop: float | None,
    take: float | None,
    by_agent: bool,
    candles: Sequence[Candle],
    now_ms: int,
) -> tuple[dict[str, Any], bool]:
    """Hindsight figures for one trade, and whether they are final (the whole hour after
    the exit has closed). `candles` are closed 1m bars covering the trade and after."""
    out: dict[str, Any] = {}
    if opened_ms is not None:
        held = [c for c in candles if c.ts + MINUTE_MS > opened_ms and c.ts < closed_ms]
        if held:
            best = max(c.high for c in held) if side == "long" else min(c.low for c in held)
            worst = min(c.low for c in held) if side == "long" else max(c.high for c in held)
            out["held_min"] = round((closed_ms - opened_ms) / MINUTE_MS, 1)
            out["best_while_open_pct"] = _signed_pct(side, entry, best)
            out["worst_while_open_pct"] = _signed_pct(side, entry, worst)

    after = [c for c in candles if c.ts >= closed_ms]  # bars that opened after the exit
    moves: dict[str, float | None] = {}
    for minutes in AFTER_EXIT_MIN:
        bar = next((c for c in after if c.ts + MINUTE_MS >= closed_ms + minutes * MINUTE_MS), None)
        moves[f"{minutes}m"] = None if bar is None else _signed_pct(side, exit_price, bar.close)
    out["after_exit_pct"] = moves

    if by_agent and (stop is not None or take is not None):
        out["if_held"] = _if_held(side, stop, take, after, closed_ms, now_ms)

    final = now_ms >= closed_ms + (HOLD_CHECK_MIN + 1) * MINUTE_MS and all(
        v is not None for v in moves.values()
    )
    return out, final


def _if_held(
    side: str,
    stop: float | None,
    take: float | None,
    after: Sequence[Candle],
    closed_ms: int,
    now_ms: int,
) -> str:
    horizon = closed_ms + HOLD_CHECK_MIN * MINUTE_MS
    for c in after:
        if c.ts >= horizon:
            break
        hit_stop = stop is not None and (c.low <= stop if side == "long" else c.high >= stop)
        hit_take = take is not None and (c.high >= take if side == "long" else c.low <= take)
        minutes = max(1, round((c.ts + MINUTE_MS - closed_ms) / MINUTE_MS))
        if hit_stop:
            return f"stop {stop:g} would have been hit {minutes} min after your exit"
        if hit_take:
            return f"take profit {take:g} would have been hit {minutes} min after your exit"
    if now_ms < horizon:
        return "neither stop nor take profit hit yet"
    return f"neither stop nor take profit hit within {HOLD_CHECK_MIN} min"
