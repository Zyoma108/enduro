"""Deterministic risk manager with veto power over every trade the agent wants to open.

The agent expresses an intent (symbol, side, stop loss, optional take profit and risk
share); the risk manager decides whether it may happen and how big it is. Sizing is
derived from the stop: the loss if the stop is hit — price distance plus round-trip
taker fees — must not exceed the per-trade risk budget. Closing a position is never
vetoed: reducing risk is always allowed.

State that must survive restarts (equity peak, start-of-day equity, kill switch, recent
opens) lives in `RiskState` and is persisted as JSON after every change.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from enduro.execution.models import InstrumentRules, Position, PositionSide

HOUR_MS = 3_600_000
# Limits trigger *at* the boundary: 1100 -> 990 is a 10% drawdown even though floating
# point computes 9.9999999...%.
_LIMIT_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class RiskLimits:
    risk_per_trade_pct: float = 1.0  # max loss at stop, % of equity
    max_leverage: float = 5.0  # max position notional / equity
    max_open_positions: int = 1
    daily_loss_limit_pct: float = 5.0  # stop opening new trades for the rest of the UTC day
    max_drawdown_pct: float = 10.0  # from equity peak: kill switch, manual reset required
    max_trades_per_hour: int = 6


@dataclass(frozen=True, slots=True)
class OpenIntent:
    symbol: str
    side: PositionSide
    stop_loss: float
    take_profit: float | None = None
    risk_pct: float | None = None  # % of equity the agent wants to risk; None = the maximum


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    reasons: tuple[str, ...]  # why it was rejected (empty when approved)
    qty: float = 0.0
    entry_price: float = 0.0  # expected fill: ask for a long, bid for a short
    notional: float = 0.0
    risk_usd: float = 0.0  # loss if the stop is hit, fees included
    risk_pct: float = 0.0
    leverage: float = 0.0  # notional / equity


def _utc_day(now_ms: int) -> str:
    return datetime.fromtimestamp(now_ms / 1000, UTC).strftime("%Y-%m-%d")


@dataclass(slots=True)
class RiskState:
    peak_equity: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    halted: str | None = None  # kill switch reason; cleared only manually
    opens: list[int] = field(default_factory=list)  # open timestamps within the last hour

    def observe_equity(self, equity: float, now_ms: int, limits: RiskLimits) -> None:
        if (day := _utc_day(now_ms)) != self.day:
            self.day, self.day_start_equity = day, equity
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = self.drawdown_pct(equity)
        if self.halted is None and drawdown >= limits.max_drawdown_pct - _LIMIT_EPS:
            self.halted = (
                f"drawdown {drawdown:.2f}% from peak {self.peak_equity:.2f} "
                f"reached the {limits.max_drawdown_pct}% limit at {_utc_day(now_ms)}"
            )

    def drawdown_pct(self, equity: float) -> float:
        return (1 - equity / self.peak_equity) * 100 if self.peak_equity > 0 else 0.0

    def daily_loss_pct(self, equity: float) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (1 - equity / self.day_start_equity) * 100

    def record_open(self, now_ms: int) -> None:
        self.opens = [t for t in self.opens if t > now_ms - HOUR_MS] + [now_ms]

    def opens_last_hour(self, now_ms: int) -> int:
        return sum(1 for t in self.opens if t > now_ms - HOUR_MS)


class RiskStateStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> RiskState:
        if not self.path.exists():
            return RiskState()
        return RiskState(**json.loads(self.path.read_text()))

    def save(self, state: RiskState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2))
        os.replace(tmp, self.path)


class RiskManager:
    def __init__(self, limits: RiskLimits, taker_fee: float, store: RiskStateStore) -> None:
        self.limits = limits
        self.taker_fee = taker_fee  # fraction per side
        self.store = store
        self.state = store.load()

    def observe_equity(self, equity: float, now_ms: int) -> None:
        self.state.observe_equity(equity, now_ms, self.limits)
        self.store.save(self.state)

    def reset_halt(self) -> None:
        """Manual kill-switch reset; the current equity becomes the new peak."""
        self.state.halted = None
        self.state.peak_equity = 0.0
        self.store.save(self.state)

    def record_open(self, now_ms: int) -> None:
        self.state.record_open(now_ms)
        self.store.save(self.state)

    def evaluate_open(
        self,
        intent: OpenIntent,
        *,
        equity: float,
        positions: list[Position],
        rules: InstrumentRules,
        bid: float,
        ask: float,
        now_ms: int,
    ) -> RiskDecision:
        self.observe_equity(equity, now_ms)
        limits, state = self.limits, self.state
        entry = ask if intent.side == "long" else bid
        reasons: list[str] = []

        def reject(reason: str) -> RiskDecision:
            return RiskDecision(False, (*reasons, reason), entry_price=entry)

        # Account-level gates.
        if state.halted:
            return reject(f"kill switch is active: {state.halted}")
        if (loss := state.daily_loss_pct(equity)) >= limits.daily_loss_limit_pct - _LIMIT_EPS:
            return reject(
                f"daily loss {loss:.2f}% reached the {limits.daily_loss_limit_pct}% limit; "
                "no new trades until 00:00 UTC"
            )
        open_positions = [p for p in positions if p.size > 0]
        if len(open_positions) >= limits.max_open_positions:
            held = ", ".join(f"{p.symbol} {p.side}" for p in open_positions)
            return reject(
                f"{len(open_positions)} open position(s) ({held}); "
                f"limit is {limits.max_open_positions}"
            )
        if (n := state.opens_last_hour(now_ms)) >= limits.max_trades_per_hour:
            return reject(
                f"{n} trades opened in the last hour; limit is {limits.max_trades_per_hour}"
            )

        # The stop (and take profit) must make sense for the side.
        if entry <= 0 or not math.isfinite(entry):
            return reject("no valid market price")
        if intent.side == "long" and not intent.stop_loss < entry:
            return reject(f"long stop loss {intent.stop_loss} must be below entry ~{entry}")
        if intent.side == "short" and not intent.stop_loss > entry:
            return reject(f"short stop loss {intent.stop_loss} must be above entry ~{entry}")
        if intent.take_profit is not None and (
            (intent.side == "long" and intent.take_profit <= entry)
            or (intent.side == "short" and intent.take_profit >= entry)
        ):
            return reject(f"take profit {intent.take_profit} is on the losing side of ~{entry}")

        # Size from the stop: loss per unit = distance to stop + taker fees both ways.
        risk_pct = limits.risk_per_trade_pct
        if intent.risk_pct is not None:
            if intent.risk_pct <= 0:
                return reject("risk_pct must be positive")
            risk_pct = min(intent.risk_pct, risk_pct)
        budget = equity * risk_pct / 100
        loss_per_unit = abs(entry - intent.stop_loss) + self.taker_fee * (entry + intent.stop_loss)
        qty = rules.round_qty(min(budget / loss_per_unit, equity * limits.max_leverage / entry))
        min_qty = rules.min_order_qty(entry)
        if qty < min_qty:
            min_risk = min_qty * loss_per_unit
            return reject(
                f"minimum order {min_qty:g} would risk {min_risk:.2f} USDT "
                f"({min_risk / equity * 100:.2f}% of equity) > budget {budget:.2f} USDT; "
                "use a closer stop or skip this trade"
            )
        risk_usd = qty * loss_per_unit
        return RiskDecision(
            approved=True,
            reasons=(),
            qty=qty,
            entry_price=entry,
            notional=qty * entry,
            risk_usd=risk_usd,
            risk_pct=risk_usd / equity * 100,
            leverage=qty * entry / equity,
        )
