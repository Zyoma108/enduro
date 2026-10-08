"""Bybit (v5, unified trading account) execution gateway on top of ccxt.

Demo trading uses the same API on api-demo.bybit.com with separate keys; ccxt switches
to it via `enable_demo_trading`. Live trading is refused unless explicitly allowed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Literal

import ccxt.async_support as ccxt

from enduro.data.ccxt_source import is_linear_usdt_perp, with_retries
from enduro.execution.base import PositionClosed
from enduro.execution.models import (
    AccountState,
    Balance,
    ClosedTrade,
    InstrumentRules,
    OrderRequest,
    OrderResult,
    Position,
    PositionSide,
)

log = logging.getLogger(__name__)

Environment = Literal["demo", "live"]

POSITION_IDX = {"long": 1, "short": 2}  # hedge mode position slots
_MARGIN_MODES = {
    "REGULAR_MARGIN": "cross",
    "ISOLATED_MARGIN": "isolated",
    "PORTFOLIO_MARGIN": "portfolio",
}
# "Not modified" answers when a setting already has the requested value.
_ALREADY_SET = ("110025", "110043", "34040")  # position mode, leverage, trading stop
_ZERO_POSITION = "zero position"  # retCode 10001: "can not set tp/sl/ts for zero position"


class LiveTradingNotAllowed(RuntimeError):
    pass


def order_params(request: OrderRequest) -> dict[str, Any]:
    """Bybit-specific params: the hedge-mode slot and reduce-only for closes."""
    params: dict[str, Any] = {"positionIdx": POSITION_IDX[request.position_side]}
    if request.action == "close":
        params["reduceOnly"] = True
    if request.client_order_id:
        params["clientOrderId"] = request.client_order_id
    # Position-level (tpslMode Full) market stop / take profit, triggered by last price.
    if request.stop_loss is not None:
        params["stopLoss"] = {"triggerPrice": request.stop_loss}
        params["slTriggerBy"] = "LastPrice"
    if request.take_profit is not None:
        params["takeProfit"] = {"triggerPrice": request.take_profit}
        params["tpTriggerBy"] = "LastPrice"
    return params


def order_from_ccxt(raw: dict[str, Any]) -> OrderResult:
    fee = raw.get("fee") or {}
    return OrderResult(
        id=str(raw["id"]),
        client_order_id=raw.get("clientOrderId"),
        symbol=raw.get("symbol") or "",
        side=raw.get("side") or "",
        status=raw.get("status") or "unknown",
        qty=float(raw.get("amount") or 0.0),
        filled=float(raw.get("filled") or 0.0),
        avg_price=float(raw["average"]) if raw.get("average") else None,
        fee=float(fee["cost"]) if fee.get("cost") is not None else None,
        ts=raw.get("timestamp"),
    )


def position_from_ccxt(raw: dict[str, Any]) -> Position | None:
    side = raw.get("side")
    if side not in ("long", "short"):
        return None

    def opt(key: str) -> float | None:
        value = raw.get(key)
        return float(value) if value not in (None, "") else None

    return Position(
        symbol=raw["symbol"],
        side=side,
        size=float(raw.get("contracts") or 0.0) * float(raw.get("contractSize") or 1.0),
        entry_price=opt("entryPrice"),
        mark_price=opt("markPrice"),
        unrealized_pnl=opt("unrealizedPnl"),
        leverage=opt("leverage"),
        liquidation_price=opt("liquidationPrice"),
        stop_loss=opt("stopLossPrice"),
        take_profit=opt("takeProfitPrice"),
    )


def closed_trade_from_bybit(raw: dict[str, Any], symbol: str) -> ClosedTrade:
    """One row of v5 /position/closed-pnl. `side` there is the closing order's side."""
    return ClosedTrade(
        order_id=raw["orderId"],
        symbol=symbol,
        side="long" if raw["side"] == "Sell" else "short",
        qty=float(raw["qty"]),
        entry_price=float(raw["avgEntryPrice"]),
        exit_price=float(raw["avgExitPrice"]),
        pnl=float(raw["closedPnl"]),
        fees=float(raw.get("openFee") or 0) + float(raw.get("closeFee") or 0),
        closed_ms=int(raw["updatedTime"]),
    )


def balance_from_ccxt(raw: dict[str, Any]) -> Balance:
    """USDT balance in USDT.

    Not the account-level `totalEquity`: Bybit reports it in USD, so it moves with the
    USDT/USD rate (~0.1%) even when nothing trades — phantom PnL for the risk layer.
    """
    accounts = ((raw.get("info") or {}).get("result") or {}).get("list") or [{}]
    coins = {c.get("coin"): c for c in accounts[0].get("coin") or []}
    usdt = raw.get("USDT") or {}
    equity = (coins.get("USDT") or {}).get("equity") or usdt.get("total") or 0.0
    return Balance(equity=float(equity), available=float(usdt.get("free") or 0.0))


def _already_set(error: Exception) -> bool:
    return any(code in str(error) for code in _ALREADY_SET)


class BybitGateway:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        environment: Environment = "demo",
        allow_live: bool = False,
    ) -> None:
        if environment == "live" and not allow_live:
            raise LiveTradingNotAllowed(
                "live trading is disabled; set execution.allow_live = true to enable it"
            )
        self.environment = environment
        self._client = ccxt.bybit(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",
                    "fetchMarkets": {"types": ["linear"]},
                    # Exchange and local clocks drift by tens of ms; signed requests
                    # outside recv_window are rejected.
                    "adjustForTimeDifference": True,
                },
            }
        )
        if environment == "demo":
            self._client.enable_demo_trading(True)

    async def _call(self, what: str, fn, *args, **kwargs):
        async def attempt():
            try:
                return await fn(*args, **kwargs)
            except ccxt.InvalidNonce:
                await self._resync_clock()
                raise  # transient: with_retries tries again with the new offset

        return await with_retries(attempt, f"bybit {what}")

    async def _resync_clock(self) -> None:
        # ccxt measures the clock offset only once, when markets load. A later clock step
        # (sleep, NTP correction) puts every signed request outside recv_window until the
        # offset is measured again.
        try:
            offset = await self._client.load_time_difference()
        except Exception as e:
            log.warning("bybit clock resync failed: %s", e)
            return
        log.warning("bybit rejected request timestamp; local clock offset now %d ms", offset)

    async def account_state(self) -> AccountState:
        info = await self._call("account info", self._client.privateGetV5AccountInfo)
        margin = _MARGIN_MODES.get((info.get("result") or {}).get("marginMode"))
        return AccountState(self.environment, margin, await self._hedge_mode())

    async def _hedge_mode(self) -> bool | None:
        # In hedge mode Bybit reports two position slots (idx 1 and 2) per symbol even
        # when flat; in one-way mode a single slot with idx 0.
        response = await self._call(
            "position list",
            self._client.privateGetV5PositionList,
            {"category": "linear", "symbol": "BTCUSDT"},
        )
        idx = {row.get("positionIdx") for row in (response.get("result") or {}).get("list", [])}
        if not idx:
            return None
        return idx >= {1, 2} or idx >= {"1", "2"}

    async def ensure_account_setup(self) -> AccountState:
        state = await self.account_state()
        if state.margin_mode != "cross":
            log.info("switching margin mode %s -> cross", state.margin_mode)
            await self._call("set margin mode", self._client.set_margin_mode, "cross")
        if not state.hedge_mode:
            log.info("switching USDT perpetuals to hedge mode")
            try:
                await self._call("set position mode", self._client.set_position_mode, True)
            except ccxt.ExchangeError as e:
                if not _already_set(e):
                    raise
        return await self.account_state()

    async def set_leverage(self, symbol: str, leverage: float) -> None:
        try:
            await self._call("set leverage", self._client.set_leverage, leverage, symbol)
        except ccxt.ExchangeError as e:
            if not _already_set(e):
                raise

    async def instrument_rules(self, symbol: str) -> InstrumentRules:
        await self._call("load markets", self._client.load_markets)
        market = self._client.market(symbol)
        if not is_linear_usdt_perp(market):
            raise ValueError(f"{symbol} is not an active USDT perpetual on bybit")
        limits = market["limits"]
        return InstrumentRules(
            symbol=symbol,
            qty_step=float(market["precision"]["amount"]),
            min_qty=float(limits["amount"]["min"] or 0.0),
            min_notional=float((limits.get("cost") or {}).get("min") or 0.0),
            price_tick=float(market["precision"]["price"]),
        )

    async def balance(self) -> Balance:
        return balance_from_ccxt(await self._call("fetch balance", self._client.fetch_balance))

    async def positions(self, symbols: Sequence[str] | None = None) -> list[Position]:
        raw = await self._call(
            "fetch positions", self._client.fetch_positions, list(symbols) if symbols else None
        )
        parsed = (position_from_ccxt(p) for p in raw)
        return [p for p in parsed if p is not None and p.size > 0]

    async def place_order(self, request: OrderRequest) -> OrderResult:
        # Not retried: a timeout does not mean the order was not placed. The caller must
        # reconcile by client_order_id instead of blindly resubmitting.
        try:
            raw = await self._client.create_order(
                request.symbol,
                request.type,
                request.side,
                request.qty,
                request.price,
                order_params(request),
            )
        except ccxt.InvalidNonce:
            await self._resync_clock()  # so the next request is signed with a valid time
            raise
        return order_from_ccxt({**raw, "symbol": raw.get("symbol") or request.symbol})

    async def set_protection(
        self,
        symbol: str,
        side: PositionSide,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        """Set (or move) the position-level stop loss / take profit. 0 removes one."""
        await self._call("load markets", self._client.load_markets)
        market = self._client.market(symbol)
        request: dict[str, Any] = {
            "category": "linear",
            "symbol": market["id"],
            "positionIdx": POSITION_IDX[side],
            "tpslMode": "Full",
        }
        if stop_loss is not None:
            request["stopLoss"] = (
                "0" if stop_loss == 0 else self._client.price_to_precision(symbol, stop_loss)
            )
            request["slTriggerBy"] = "LastPrice"
        if take_profit is not None:
            request["takeProfit"] = (
                "0" if take_profit == 0 else self._client.price_to_precision(symbol, take_profit)
            )
            request["tpTriggerBy"] = "LastPrice"
        try:
            await self._call("trading stop", self._client.privatePostV5PositionTradingStop, request)
        except ccxt.ExchangeError as e:
            if _ZERO_POSITION in str(e):
                raise PositionClosed(f"{side} position on {symbol} is already closed") from e
            if not _already_set(e):
                raise

    async def fetch_order(self, order_id: str, symbol: str) -> OrderResult:
        raw = await self._call(
            "fetch order", self._client.fetch_order, order_id, symbol, {"acknowledged": True}
        )
        return order_from_ccxt(raw)

    async def wait_for_fill(
        self, order_id: str, symbol: str, timeout_s: float = 10.0
    ) -> OrderResult:
        """Poll until the order is no longer open (or the timeout passes)."""
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            order = await self.fetch_order(order_id, symbol)
            if order.status != "open" or asyncio.get_running_loop().time() > deadline:
                return order
            await asyncio.sleep(0.3)

    async def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        raw = await self._call("cancel order", self._client.cancel_order, order_id, symbol)
        return order_from_ccxt({**raw, "symbol": raw.get("symbol") or symbol})

    async def open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        raw = await self._call("open orders", self._client.fetch_open_orders, symbol)
        return [order_from_ccxt(o) for o in raw]

    async def closed_trades(self, limit: int = 10) -> list[ClosedTrade]:
        await self._call("load markets", self._client.load_markets)
        raw = await self._call(
            "closed pnl",
            self._client.privateGetV5PositionClosedPnl,
            {"category": "linear", "limit": limit},
        )
        return [
            closed_trade_from_bybit(row, self._client.safe_symbol(row["symbol"], None, "", "swap"))
            for row in raw["result"]["list"]
        ]

    async def close(self) -> None:
        await self._client.close()
