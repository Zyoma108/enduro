"""The only path from an agent's intent to the exchange.

open:  risk decision -> leverage -> market order with the stop attached -> verify the
       stop is on the exchange (if not: set it, and if that fails, close at once) -> journal
close: market reduce-only for the whole position -> journal
protect: risk check -> move stop / take profit on the exchange -> journal

Results are plain dicts so they can go straight back to the agent as tool results.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable
from typing import Any

from enduro.core.models import now_ms
from enduro.execution.base import ExecutionGateway
from enduro.execution.models import OrderRequest, Position, PositionSide
from enduro.journal.journal import Journal
from enduro.risk.manager import OpenIntent, RiskManager

log = logging.getLogger(__name__)

Quote = Callable[[str], Awaitable[tuple[float, float]]]  # symbol -> (bid, ask)


class TradingService:
    def __init__(
        self,
        gateway: ExecutionGateway,
        risk: RiskManager,
        journal: Journal,
        quote: Quote,
        dry_run: bool = False,
    ) -> None:
        self.gateway = gateway
        self.risk = risk
        self.journal = journal
        self.quote = quote
        self.dry_run = dry_run  # evaluate and journal, but never send orders
        self._leverage_set: set[str] = set()

    async def account(self) -> dict[str, Any]:
        balance = await self.gateway.balance()
        positions = await self.gateway.positions()
        self.risk.observe_equity(balance.equity, now_ms())
        state, limits = self.risk.state, self.risk.limits
        return {
            "environment": self.gateway.environment + (" (dry run)" if self.dry_run else ""),
            "equity_usdt": round(balance.equity, 2),
            "available_usdt": round(balance.available, 2),
            "positions": [_position_view(p) for p in positions],
            "risk": {
                "kill_switch": state.halted,
                "daily_loss_pct": round(state.daily_loss_pct(balance.equity), 2),
                "daily_loss_limit_pct": limits.daily_loss_limit_pct,
                "drawdown_from_peak_pct": round(state.drawdown_pct(balance.equity), 2),
                "max_drawdown_pct": limits.max_drawdown_pct,
                "trades_last_hour": state.opens_last_hour(now_ms()),
                "max_trades_per_hour": limits.max_trades_per_hour,
                "risk_per_trade_pct": limits.risk_per_trade_pct,
                "max_leverage": limits.max_leverage,
            },
        }

    async def open(self, intent: OpenIntent, thesis: str) -> dict[str, Any]:
        balance = await self.gateway.balance()
        positions = await self.gateway.positions()
        rules = await self.gateway.instrument_rules(intent.symbol)
        bid, ask = await self.quote(intent.symbol)
        decision = self.risk.evaluate_open(
            intent,
            equity=balance.equity,
            positions=positions,
            rules=rules,
            bid=bid,
            ask=ask,
            now_ms=now_ms(),
        )
        self.journal.write("risk", intent=intent, decision=decision, thesis=thesis)
        if not decision.approved:
            return {"opened": False, "rejected_by_risk": list(decision.reasons)}
        plan = {
            "qty": decision.qty,
            "expected_entry": decision.entry_price,
            "notional_usdt": round(decision.notional, 2),
            "risk_usdt": round(decision.risk_usd, 2),
            "risk_pct": round(decision.risk_pct, 3),
            "leverage": round(decision.leverage, 2),
        }
        if self.dry_run:
            self.risk.record_open(now_ms())
            self.journal.write("order", action="open", dry_run=True, intent=intent, plan=plan)
            return {"opened": False, "dry_run": True, "would_open": plan}

        if intent.symbol not in self._leverage_set:
            await self.gateway.set_leverage(intent.symbol, math.ceil(self.risk.limits.max_leverage))
            self._leverage_set.add(intent.symbol)
        request = OrderRequest(
            intent.symbol,
            intent.side,
            "open",
            decision.qty,
            client_order_id=f"enduro-{now_ms()}-o",
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
        )
        placed = await self.gateway.place_order(request)
        order = await self.gateway.wait_for_fill(placed.id, intent.symbol)
        self.risk.record_open(now_ms())
        self.journal.write("order", action="open", request=request, result=order, plan=plan)
        if order.filled <= 0:
            return {"opened": False, "error": f"order not filled (status {order.status})"}

        position = await self._position(intent.symbol, intent.side)
        protected = await self._ensure_stop(intent, position)
        return {
            "opened": True,
            "side": intent.side,
            "qty": order.filled,
            "avg_price": order.avg_price,
            "fee_usdt": order.fee,
            "stop_loss": position.stop_loss if position else None,
            "take_profit": position.take_profit if position else None,
            "stop_verified": protected,
            **{k: v for k, v in plan.items() if k in ("risk_usdt", "risk_pct", "leverage")},
        }

    async def close(self, symbol: str, side: PositionSide, reason: str) -> dict[str, Any]:
        position = await self._position(symbol, side)
        if position is None:
            return {"closed": False, "error": f"no open {side} position on {symbol}"}
        if self.dry_run:
            self.journal.write("order", action="close", dry_run=True, symbol=symbol, side=side)
            return {"closed": False, "dry_run": True}
        request = OrderRequest(
            symbol, side, "close", position.size, client_order_id=f"enduro-{now_ms()}-c"
        )
        placed = await self.gateway.place_order(request)
        order = await self.gateway.wait_for_fill(placed.id, symbol)
        self.journal.write("order", action="close", request=request, result=order, reason=reason)
        pnl = None
        if order.avg_price and position.entry_price:
            sign = 1 if side == "long" else -1
            pnl = sign * (order.avg_price - position.entry_price) * order.filled
        return {
            "closed": order.filled >= position.size,
            "qty": order.filled,
            "avg_price": order.avg_price,
            "entry_price": position.entry_price,
            "gross_pnl_usdt": None if pnl is None else round(pnl, 4),
            "close_fee_usdt": order.fee,
        }

    async def protect(
        self,
        symbol: str,
        side: PositionSide,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> dict[str, Any]:
        position = await self._position(symbol, side)
        if position is None:
            return {"updated": False, "error": f"no open {side} position on {symbol}"}
        balance = await self.gateway.balance()
        bid, ask = await self.quote(symbol)
        ok, reason = self.risk.evaluate_protection(
            position,
            stop_loss=stop_loss,
            take_profit=take_profit,
            mark=(bid + ask) / 2,
            equity=balance.equity,
        )
        self.journal.write(
            "risk",
            protection={"stop_loss": stop_loss, "take_profit": take_profit},
            ok=ok,
            reason=reason,
            position=position,
        )
        if not ok:
            return {"updated": False, "rejected_by_risk": reason}
        if self.dry_run:
            return {"updated": False, "dry_run": True}
        await self.gateway.set_protection(symbol, side, stop_loss, take_profit)
        updated = await self._position(symbol, side)
        return {"updated": True, "position": _position_view(updated) if updated else None}

    async def _position(self, symbol: str, side: PositionSide) -> Position | None:
        return next((p for p in await self.gateway.positions([symbol]) if p.side == side), None)

    async def _ensure_stop(self, intent: OpenIntent, position: Position | None) -> bool:
        """Never leave a position without its stop on the exchange."""
        if position is None or position.stop_loss is not None:
            return position is not None
        log.warning("stop missing on %s %s after fill, setting it", intent.symbol, intent.side)
        try:
            await self.gateway.set_protection(
                intent.symbol, intent.side, intent.stop_loss, intent.take_profit
            )
            fixed = await self._position(intent.symbol, intent.side)
            if fixed and fixed.stop_loss is not None:
                return True
        except Exception:
            log.exception("could not set stop on %s %s", intent.symbol, intent.side)
        self.journal.write("error", what="stop missing after open; closing", intent=intent)
        await self.close(intent.symbol, intent.side, reason="stop could not be placed")
        return False


def _position_view(p: Position) -> dict[str, Any]:
    return {
        "symbol": p.symbol,
        "side": p.side,
        "size": p.size,
        "entry_price": p.entry_price,
        "mark_price": p.mark_price,
        "unrealized_pnl_usdt": p.unrealized_pnl,
        "stop_loss": p.stop_loss,
        "take_profit": p.take_profit,
        "liquidation_price": p.liquidation_price,
    }
