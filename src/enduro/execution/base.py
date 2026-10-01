"""Exchange-agnostic execution interface. The risk layer and agent depend only on this."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from enduro.execution.models import (
    AccountState,
    Balance,
    InstrumentRules,
    OrderRequest,
    OrderResult,
    Position,
)


class ExecutionGateway(Protocol):
    environment: str  # "demo" | "live"

    async def account_state(self) -> AccountState: ...

    async def ensure_account_setup(self) -> AccountState:
        """Switch the account to cross margin and hedge mode if it is not already."""
        ...

    async def set_leverage(self, symbol: str, leverage: float) -> None: ...

    async def instrument_rules(self, symbol: str) -> InstrumentRules: ...

    async def balance(self) -> Balance: ...

    async def positions(self, symbols: Sequence[str] | None = None) -> list[Position]:
        """Open (non-zero) positions, optionally limited to `symbols`."""
        ...

    async def place_order(self, request: OrderRequest) -> OrderResult: ...

    async def fetch_order(self, order_id: str, symbol: str) -> OrderResult: ...

    async def cancel_order(self, order_id: str, symbol: str) -> OrderResult: ...

    async def open_orders(self, symbol: str | None = None) -> list[OrderResult]: ...

    async def close(self) -> None: ...
