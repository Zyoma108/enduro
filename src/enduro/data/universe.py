"""The set of symbols the radar scans.

A symbol qualifies when it is a USDT perpetual listed on both the reference exchange
(source of truth) and the execution exchange, both exchanges agree on its asset class
(crypto by default — tokenized stocks, commodities etc. are excluded), and it is liquid
enough on the execution exchange to trade without slippage eating the profit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from dataclasses import dataclass

from enduro.data.base import MarketDataSource


@dataclass(frozen=True, slots=True)
class UniverseEntry:
    symbol: str
    quote_volume_24h: float  # on the execution exchange, USDT


def common_symbols(
    reference: dict[str, str], execution: dict[str, str], asset_classes: Collection[str]
) -> list[str]:
    """Symbols listed on both exchanges whose asset class is allowed on both."""
    return sorted(
        s
        for s, cls in execution.items()
        if cls in asset_classes and reference.get(s) in asset_classes
    )


def select_universe(
    symbols: Collection[str], execution_volumes: dict[str, float], min_quote_volume: float
) -> list[UniverseEntry]:
    """Symbols liquid enough on the execution exchange, most traded first."""
    entries = [
        UniverseEntry(s, execution_volumes[s])
        for s in symbols
        if execution_volumes.get(s, 0.0) >= min_quote_volume
    ]
    return sorted(entries, key=lambda e: e.quote_volume_24h, reverse=True)


async def build_universe(
    reference: MarketDataSource,
    execution: MarketDataSource,
    min_quote_volume: float,
    asset_classes: Collection[str] = ("crypto",),
) -> list[UniverseEntry]:
    reference_markets, execution_markets = await asyncio.gather(
        reference.list_linear_usdt_perps(), execution.list_linear_usdt_perps()
    )
    common = common_symbols(reference_markets, execution_markets, asset_classes)
    volumes = await execution.fetch_quote_volumes(common)
    return select_universe(common, volumes, min_quote_volume)
