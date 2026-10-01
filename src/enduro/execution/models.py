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
    client_order_id: str | None = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("qty must be positive")
        if self.type == "limit" and self.price is None:
            raise ValueError("limit order requires a price")

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


@dataclass(frozen=True, slots=True)
class Balance:
    equity: float  # USDT, including unrealized PnL
    available: float  # USDT free for new positions


@dataclass(frozen=True, slots=True)
class AccountState:
    environment: str  # "demo" | "live"
    margin_mode: str | None  # "cross" | "isolated" | "portfolio"
    hedge_mode: bool | None  # None if it could not be determined
