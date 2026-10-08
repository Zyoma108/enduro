"""Execution-side models: what we ask the exchange to do and what it reports back.

Positions run in hedge mode: a symbol can have a long and a short position at the same
time, so a position is identified by (symbol, side), and every order says which of the
two it opens or closes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Literal

PositionSide = Literal["long", "short"]
OrderAction = Literal["open", "close"]
OrderType = Literal["market", "limit"]


@dataclass(frozen=True, slots=True)
class InstrumentRules:
    symbol: str
    qty_step: float  # order quantity must be a multiple of this (base currency)
    min_qty: float
    min_notional: float  # minimum order value, USDT
    price_tick: float

    def round_qty(self, qty: float) -> float:
        """Round down to the quantity step (never order more than asked)."""
        step = Decimal(str(self.qty_step))
        return float((Decimal(str(qty)) / step).to_integral_value(ROUND_DOWN) * step)

    def min_order_qty(self, price: float) -> float:
        """Smallest valid quantity at `price` that satisfies both min qty and min notional."""
        steps = math.ceil(max(self.min_qty, self.min_notional / price) / self.qty_step - 1e-9)
        return self.round_qty(steps * self.qty_step)


@dataclass(frozen=True, slots=True)
class OrderRequest:
    symbol: str
    position_side: PositionSide
    action: OrderAction
    qty: float  # base currency
    type: OrderType = "market"
    price: float | None = None  # required for limit orders
    post_only: bool = False  # limit only: cancelled instead of filling as a taker
    client_order_id: str | None = None
    # Attached to the position on the exchange the moment the order fills (opens only).
    stop_loss: float | None = None
    take_profit: float | None = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("qty must be positive")
        if self.type == "limit" and self.price is None:
            raise ValueError("limit order requires a price")
        if self.post_only and self.type != "limit":
            raise ValueError("post-only applies to limit orders")
        if self.action == "close" and (self.stop_loss or self.take_profit):
            raise ValueError("stop loss / take profit can only be attached to an open")

    @property
    def side(self) -> Literal["buy", "sell"]:
        """Exchange order side: opening long or closing short buys, the rest sells."""
        buys = (self.position_side == "long") == (self.action == "open")
        return "buy" if buys else "sell"


@dataclass(frozen=True, slots=True)
class OrderResult:
    id: str
    client_order_id: str | None
    symbol: str
    side: str
    status: str  # ccxt unified: open, closed, canceled, rejected, expired
    qty: float
    filled: float
    avg_price: float | None
    fee: float | None  # USDT
    ts: int | None
    price: float | None = None  # limit price (None for market orders)


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    side: PositionSide
    size: float  # base currency, 0 when flat
    entry_price: float | None
    mark_price: float | None
    unrealized_pnl: float | None
    leverage: float | None
    liquidation_price: float | None
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass(frozen=True, slots=True)
class Balance:
    equity: float  # USDT, including unrealized PnL
    available: float  # USDT free for new positions


@dataclass(frozen=True, slots=True)
class AccountState:
    environment: str  # "demo" | "live"
    margin_mode: str | None  # "cross" | "isolated" | "portfolio"
    hedge_mode: bool | None  # None if it could not be determined


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """A position (or part of one) closed, as the exchange books it — whoever closed it:
    our own close order, the stop loss, the take profit or a liquidation."""

    order_id: str  # the closing order
    symbol: str
    side: PositionSide  # the position that was closed
    qty: float
    entry_price: float
    exit_price: float
    pnl: float  # USDT, net of the opening and closing fees
    fees: float
    closed_ms: int
