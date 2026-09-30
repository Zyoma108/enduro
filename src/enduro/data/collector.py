"""Runs trade and order book streams for every source and publishes them to the bus."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Sequence

from enduro.core.models import MarketEvent
from enduro.data.base import MarketDataSource
from enduro.data.bus import EventBus

log = logging.getLogger(__name__)

StreamFactory = Callable[[], AsyncIterator[MarketEvent]]


class Collector:
    def __init__(
        self,
        sources: Sequence[MarketDataSource],
        symbols: Sequence[str],
        bus: EventBus,
        book_depth: int = 20,
        max_backoff_s: float = 30.0,
    ) -> None:
        self._sources = sources
        self._symbols = list(symbols)
        self._bus = bus
        self._book_depth = book_depth
        self._max_backoff_s = max_backoff_s

    async def run(self) -> None:
        try:
            async with asyncio.TaskGroup() as tg:
                for src in self._sources:
                    tg.create_task(
                        self._supervise(
                            f"{src.exchange}:trades",
                            lambda src=src: src.stream_trades(self._symbols),
                        )
                    )
                    tg.create_task(
                        self._supervise(
                            f"{src.exchange}:books",
                            lambda src=src: src.stream_order_books(self._symbols, self._book_depth),
                        )
                    )
        finally:
            await asyncio.gather(*(src.close() for src in self._sources), return_exceptions=True)

    async def _supervise(self, name: str, open_stream: StreamFactory) -> None:
        """Keep a stream alive forever, reconnecting with exponential backoff on errors."""
        backoff = 1.0
        while True:
            try:
                async for event in open_stream():
                    self._bus.publish(event)
                    backoff = 1.0
                log.warning("stream %s ended, restarting", name)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("stream %s failed, retrying in %.0fs", name, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff_s)
