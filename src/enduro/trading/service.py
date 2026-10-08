"""The only path from an agent's intent to the exchange.

open:  risk decision (sized at the chase bound) -> leverage -> post-only limit order with
       the stop attached, chasing the best bid/ask until filled, timed out, or price runs
       past the bound (then cancelled: nothing opens) -> verify the stop is on the exchange
       (if not: set it, and if that fails, close at once) -> take profit as a resting
       reduce-only limit order -> journal
close: cancel the take profit -> post-only reduce-only limit chasing the touch for a few
       seconds -> market for whatever is left (or market at once when urgent) -> journal
protect: risk check -> move the stop on the exchange, move / place / cancel the take
       profit order -> journal

Limit orders pay the maker fee instead of the taker fee. The stop stays a market order
on the exchange: getting out matters more than the fee.

Results are plain dicts so they can go straight back to the agent as tool results.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from enduro.core.models import now_ms
from enduro.execution.base import ExecutionGateway, OrderNotOpen, PositionClosed
from enduro.execution.models import InstrumentRules, OrderRequest, Position, PositionSide
from enduro.journal.journal import Journal
from enduro.risk.manager import OpenIntent, RiskManager

log = logging.getLogger(__name__)

Quote = Callable[[str], Awaitable[tuple[float, float]]]  # symbol -> (bid, ask)

TAKE_PROFIT_TAG = "-tp"  # client order id suffix of our take profit orders


@dataclass(frozen=True, slots=True)
class ChaseSettings:
    open_s: float = 20.0  # how long an entry chases before it is cancelled
    # How far past the touch an entry may chase, as a share of the stop distance.
    open_max_stop_share: float = 0.1
    close_s: float = 15.0  # how long a close chases before the rest goes at market
    poll_s: float = 1.0  # how often the order is checked and re-priced


@dataclass(slots=True)
class ChaseResult:
    filled: float = 0.0
    cost: float = 0.0  # sum of price * qty over fills
    fee: float = 0.0
    orders: int = 0
    reprices: int = 0
    outcome: str = "timeout"  # filled | timeout | ran_away | error
    error: str | None = None
    order_ids: list[str] = field(default_factory=list)  # exchange ids of every order placed

    @property
    def avg_price(self) -> float | None:
        return self.cost / self.filled if self.filled else None

    def summary(self, seconds: float) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "orders": self.orders,
            "reprices": self.reprices,
            "seconds": round(seconds, 1),
            **({"error": self.error} if self.error else {}),
        }


class TradingService:
    def __init__(
        self,
        gateway: ExecutionGateway,
        risk: RiskManager,
        journal: Journal,
        quote: Quote,
        dry_run: bool = False,
        chase: ChaseSettings | None = None,
    ) -> None:
        self.gateway = gateway
        self.risk = risk
        self.journal = journal
        self.quote = quote
        self.dry_run = dry_run  # evaluate and journal, but never send orders
        self.chase = chase or ChaseSettings()
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
            "positions": [await self._position_view(p) for p in positions],
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

    # ------------------------------------------------------------------ open

    async def open(self, intent: OpenIntent, thesis: str) -> dict[str, Any]:
        balance = await self.gateway.balance()
        positions = await self.gateway.positions()
        rules = await self.gateway.instrument_rules(intent.symbol)
        bid, ask = await self.quote(intent.symbol)
        bound = self._entry_bound(intent, bid, ask, rules)
        decision = self.risk.evaluate_open(
            intent,
            equity=balance.equity,
            positions=positions,
            rules=rules,
            bid=bid,
            ask=ask,
            now_ms=now_ms(),
            worst_entry=bound,
        )
        self.journal.write("risk", intent=intent, decision=decision, thesis=thesis)
        if not decision.approved:
            return {"opened": False, "rejected_by_risk": list(decision.reasons)}
        plan = {
            "qty": decision.qty,
            "limit_from": bid if intent.side == "long" else ask,
            "chase_bound": bound,
            "notional_usdt": round(decision.notional, 2),
            "risk_usdt": round(decision.risk_usd, 2),  # at the bound: the worst case
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
        await self._cancel_take_profit(intent.symbol, intent.side)  # stale, from a stop-out
        request = OrderRequest(
            intent.symbol,
            intent.side,
            "open",
            decision.qty,
            type="limit",
            price=plan["limit_from"],
            post_only=True,
            client_order_id=f"enduro-{now_ms()}-o",
            stop_loss=intent.stop_loss,
        )
        started = asyncio.get_running_loop().time()
        chase = await self._chase(request, rules, bound=bound, seconds=self.chase.open_s)
        seconds = asyncio.get_running_loop().time() - started
        # The exchange position is the truth about what filled (an order may have filled
        # between our last check and its cancellation).
        position = await self._position(intent.symbol, intent.side)
        if position is None and chase.filled > 0:
            # Filled, and the exchange stop already closed it (a stop inside the noise).
            self.risk.record_open(now_ms())
            result = {
                "id": request.client_order_id,
                "status": "filled",
                "qty": decision.qty,
                "filled": chase.filled,
                "avg_price": chase.avg_price,
                "fee": round(chase.fee, 6),
                "ts": now_ms(),
            }
            self.journal.write(
                "order",
                action="open",
                request=request,
                result=result,
                plan=plan,
                take_profit=intent.take_profit,
                chase=chase.summary(seconds),
            )
            return {
                "opened": True,
                "already_closed": True,
                "qty": chase.filled,
                "avg_price": chase.avg_price,
                "note": "the entry filled and the position is already gone: the stop on the "
                "exchange closed it at once (the stop was inside the price noise)",
            }
        if position is None:
            self.journal.write(
                "order",
                action="open_cancelled",
                request=request,
                plan=plan,
                chase=chase.summary(seconds),
            )
            why = {
                "ran_away": "price moved past the chase bound",
                "timeout": f"not filled within {self.chase.open_s:.0f} s",
                "error": f"order error: {chase.error}",
            }.get(chase.outcome, chase.outcome)
            return {
                "opened": False,
                "not_filled": why,
                "chase_bound": bound,
                "note": "nothing was opened; the entry limit order was cancelled",
            }

        self.risk.record_open(now_ms())
        result = {
            "id": request.client_order_id,
            "status": "filled" if position.size >= decision.qty else "partially filled",
            "qty": decision.qty,
            "filled": position.size,
            "avg_price": position.entry_price or chase.avg_price,
            "fee": round(chase.fee, 6),
            "ts": now_ms(),
        }
        self.journal.write(
            "order",
            action="open",
            request=request,
            result=result,
            plan=plan,
            take_profit=intent.take_profit,
            chase=chase.summary(seconds),
        )
        protected = await self._ensure_stop(intent, position)
        take_profit_error = None
        if protected and intent.take_profit:
            take_profit_error = await self._set_take_profit(
                intent.symbol, intent.side, intent.take_profit, position.size
            )
        view = await self._position_view(position) if protected else None
        return {
            "opened": True,
            "side": intent.side,
            "qty": position.size,
            "requested_qty": decision.qty,
            "avg_price": result["avg_price"],
            "fee_usdt": result["fee"],
            "entry": "maker (post-only limit)",
            "stop_loss": position.stop_loss,
            "take_profit": view["take_profit"] if view else None,
            **({"take_profit_error": take_profit_error} if take_profit_error else {}),
            "stop_verified": protected,
            **{k: v for k, v in plan.items() if k in ("risk_usdt", "risk_pct", "leverage")},
        }

    def _entry_bound(
        self, intent: OpenIntent, bid: float, ask: float, rules: InstrumentRules
    ) -> float:
        """The worst price the entry may chase to: a share of the stop distance past the
        touch, so a runaway move is not chased and the risk stays within budget."""
        share = self.chase.open_max_stop_share
        tick = rules.price_tick
        if intent.side == "long":
            bound = ask + share * max(0.0, ask - intent.stop_loss)
            return round(math.floor(bound / tick + 1e-9) * tick, 12)
        bound = bid - share * max(0.0, intent.stop_loss - bid)
        return round(math.ceil(bound / tick - 1e-9) * tick, 12)

    # ----------------------------------------------------------------- close

    async def close(
        self, symbol: str, side: PositionSide, reason: str, urgent: bool = False
    ) -> dict[str, Any]:
        position = await self._position(symbol, side)
        if position is None:
            return {"closed": False, "error": f"no open {side} position on {symbol}"}
        if self.dry_run:
            self.journal.write("order", action="close", dry_run=True, symbol=symbol, side=side)
            return {"closed": False, "dry_run": True}
        # A resting take profit would compete with the close for the same position.
        await self._cancel_take_profit(symbol, side)
        tag = f"enduro-{now_ms()}-c"
        bid, ask = await self.quote(symbol)
        limit = OrderRequest(
            symbol,
            side,
            "close",
            position.size,
            type="limit",
            price=ask if side == "long" else bid,
            post_only=True,
            client_order_id=tag,
        )
        started = asyncio.get_running_loop().time()
        chase = ChaseResult(outcome="skipped")
        if not urgent:
            rules = await self.gateway.instrument_rules(symbol)
            chase = await self._chase(limit, rules, bound=None, seconds=self.chase.close_s)
        rest = await self._position(symbol, side)
        market_fill = None
        if rest is not None:
            request = OrderRequest(symbol, side, "close", rest.size, client_order_id=f"{tag}m")
            placed = await self.gateway.place_order(request)
            chase.order_ids.append(placed.id)
            market_fill = await self.gateway.wait_for_fill(placed.id, symbol)
            chase.filled += market_fill.filled
            chase.cost += market_fill.filled * (market_fill.avg_price or 0.0)
            chase.fee += market_fill.fee or 0.0
        seconds = asyncio.get_running_loop().time() - started
        left = await self._position(symbol, side)
        avg = chase.avg_price
        pnl = None
        if avg and position.entry_price:
            sign = 1 if side == "long" else -1
            pnl = sign * (avg - position.entry_price) * chase.filled
        maker_qty = chase.filled - (market_fill.filled if market_fill else 0.0)
        self.journal.write(
            "order",
            action="close",
            request=replace(limit, type="market", price=None, post_only=False) if urgent else limit,
            result={
                "id": tag,
                # The exchange books the close under one of these: the trade sync matches
                # its closing order id against them to tell our exits from stops.
                "order_ids": chase.order_ids,
                "status": "closed" if left is None else "partially closed",
                "qty": position.size,
                "filled": chase.filled,
                "avg_price": avg,
                "fee": round(chase.fee, 6),
                "ts": now_ms(),
            },
            reason=reason,
            urgent=urgent,
            maker_qty=maker_qty,
            chase=chase.summary(seconds),
            entry_price=position.entry_price,
            gross_pnl_usdt=None if pnl is None else round(pnl, 4),
        )
        return {
            "closed": left is None,
            "qty": chase.filled,
            "avg_price": avg,
            "entry_price": position.entry_price,
            "gross_pnl_usdt": None if pnl is None else round(pnl, 4),
            "close_fee_usdt": round(chase.fee, 6),
            "maker_qty": maker_qty,
            "market_qty": market_fill.filled if market_fill else 0.0,
        }

    # --------------------------------------------------------------- protect

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
        closed = {
            "updated": False,
            "error": f"the {side} position on {symbol} is already closed: its stop or "
            "take profit filled on the exchange",
        }
        if stop_loss is not None:
            try:
                await self.gateway.set_protection(symbol, side, stop_loss=stop_loss)
            except PositionClosed:
                return closed  # raced with the exchange: the stop or take profit filled
        take_profit_error = None
        if take_profit == 0:
            await self._cancel_take_profit(symbol, side)
        elif take_profit is not None:
            take_profit_error = await self._set_take_profit(
                symbol, side, take_profit, position.size
            )
        updated = await self._position(symbol, side)
        if updated is None:
            return closed
        return {
            "updated": take_profit_error is None,
            **({"take_profit_error": take_profit_error} if take_profit_error else {}),
            "position": await self._position_view(updated),
        }

    # ------------------------------------------------------------- internals

    async def _chase(
        self,
        template: OrderRequest,
        rules: InstrumentRules,
        *,
        bound: float | None,
        seconds: float,
    ) -> ChaseResult:
        """Work a post-only limit order at the passive touch (bid to buy, ask to sell),
        re-pricing it as the touch moves, until it fills, `seconds` pass, or the touch
        moves past `bound`. A post-only order that would have crossed is cancelled by
        the exchange; then a fresh one is placed a tick behind the touch (two after
        repeated rejections): Bybit demo matches against the last trade, so an order at
        the touch is rejected whenever the last trade printed there."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        buy = template.side == "buy"
        result = ChaseResult()
        order_id: str | None = None
        order_price = 0.0
        rejections = 0
        tick = rules.price_tick
        min_qty = rules.min_order_qty(template.price or 1.0)

        def account(filled: float, avg: float | None, fee: float | None) -> None:
            result.filled += filled
            result.cost += filled * (avg or 0.0)
            result.fee += fee or 0.0

        try:
            while True:
                if order_id is not None:
                    current = await self.gateway.fetch_order(order_id, template.symbol)
                    if current.status != "open":  # filled, or post-only cancelled
                        account(current.filled, current.avg_price, current.fee)
                        if current.filled < current.qty:
                            rejections += 1
                        order_id = None
                remaining = rules.round_qty(template.qty - result.filled)
                if remaining < min_qty or remaining <= 0:
                    result.outcome = "filled"
                    break
                if loop.time() >= deadline:
                    break
                bid, ask = await self.quote(template.symbol)
                behind = min(rejections, 2) * tick
                price = round((bid - behind) if buy else (ask + behind), 12)
                if bound is not None and (price > bound if buy else price < bound):
                    result.outcome = "ran_away"
                    break
                if order_id is None:
                    result.orders += 1
                    placed = await self.gateway.place_order(
                        replace(
                            template,
                            qty=remaining,
                            price=price,
                            client_order_id=f"{template.client_order_id}{result.orders}",
                        )
                    )
                    order_id, order_price = placed.id, price
                    result.order_ids.append(placed.id)
                elif price != order_price:
                    try:
                        await self.gateway.amend_order(
                            order_id, template.symbol, template.side, price
                        )
                        order_price = price
                        result.reprices += 1
                    except OrderNotOpen:
                        pass  # filled or cancelled meanwhile: the next fetch accounts it
                    except Exception as e:  # e.g. the amended price would cross: keep it
                        log.warning("amend %s to %s failed: %s", order_id, price, e)
                await asyncio.sleep(self.chase.poll_s)
        except Exception as e:  # never leave a chase order behind; the caller reconciles
            log.exception("chase of %s failed", template.client_order_id)
            result.outcome, result.error = "error", f"{type(e).__name__}: {e}"
        if order_id is not None:
            try:
                await self.gateway.cancel_order(order_id, template.symbol)
            except OrderNotOpen:
                pass
            except Exception:
                log.exception("could not cancel chase order %s", order_id)
            try:
                final = await self.gateway.fetch_order(order_id, template.symbol)
                account(final.filled, final.avg_price, final.fee)
            except Exception:
                log.exception("could not read chase order %s", order_id)
        return result

    async def _take_profit_orders(self, symbol: str, side: PositionSide) -> list:
        closing = "sell" if side == "long" else "buy"
        return [
            o
            for o in await self.gateway.open_orders(symbol)
            if (o.client_order_id or "").endswith(TAKE_PROFIT_TAG) and o.side == closing
        ]

    async def _cancel_take_profit(self, symbol: str, side: PositionSide) -> None:
        for order in await self._take_profit_orders(symbol, side):
            try:
                await self.gateway.cancel_order(order.id, symbol)
            except OrderNotOpen:
                pass

    async def _set_take_profit(
        self, symbol: str, side: PositionSide, price: float, qty: float
    ) -> str | None:
        """Rest a reduce-only limit order at the take profit (maker fee), moving the
        existing one if there is one. Returns an error text instead of raising: the
        position keeps its stop either way."""
        try:
            existing = await self._take_profit_orders(symbol, side)
            if existing and existing[0].qty == qty:
                closing = "sell" if side == "long" else "buy"
                await self.gateway.amend_order(existing[0].id, symbol, closing, price)
                for extra in existing[1:]:
                    await self.gateway.cancel_order(extra.id, symbol)
                return None
            await self._cancel_take_profit(symbol, side)
            await self.gateway.place_order(
                OrderRequest(
                    symbol,
                    side,
                    "close",
                    qty,
                    type="limit",
                    price=price,
                    client_order_id=f"enduro-{now_ms()}{TAKE_PROFIT_TAG}",
                )
            )
            return None
        except Exception as e:
            log.exception("take profit %s on %s %s failed", price, symbol, side)
            return f"take profit not placed: {type(e).__name__}: {e}"

    async def _position(self, symbol: str, side: PositionSide) -> Position | None:
        return next((p for p in await self.gateway.positions([symbol]) if p.side == side), None)

    async def _position_view(self, p: Position) -> dict[str, Any]:
        take_profit = p.take_profit  # a position-level one, if any
        try:
            orders = await self._take_profit_orders(p.symbol, p.side)
        except Exception:
            log.warning("cannot read take profit orders for %s", p.symbol, exc_info=True)
            orders = []
        limit_price = next((o.price for o in orders if o.price), None)
        return {
            "symbol": p.symbol,
            "side": p.side,
            "size": p.size,
            "entry_price": p.entry_price,
            "mark_price": p.mark_price,
            "unrealized_pnl_usdt": p.unrealized_pnl,
            "stop_loss": p.stop_loss,
            "take_profit": limit_price or take_profit,
            "liquidation_price": p.liquidation_price,
        }

    async def _ensure_stop(self, intent: OpenIntent, position: Position | None) -> bool:
        """Never leave a position without its stop on the exchange."""
        if position is None or position.stop_loss is not None:
            return position is not None
        log.warning("stop missing on %s %s after fill, setting it", intent.symbol, intent.side)
        try:
            await self.gateway.set_protection(intent.symbol, intent.side, intent.stop_loss)
            fixed = await self._position(intent.symbol, intent.side)
            if fixed and fixed.stop_loss is not None:
                return True
        except Exception:
            log.exception("could not set stop on %s %s", intent.symbol, intent.side)
        self.journal.write("error", what="stop missing after open; closing", intent=intent)
        await self.close(intent.symbol, intent.side, reason="stop could not be placed", urgent=True)
        return False
