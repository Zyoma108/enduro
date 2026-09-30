"""Runs trade and order book streams for every source and publishes them to the bus.

The symbol set can be changed at runtime with `set_symbols` (the focus follows whatever
coin the agent is working): streams restart with the new set and symbols that left it
are unsubscribed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

from enduro.core.models import MarketEvent
from enduro.data.base import MarketDataSource
from enduro.data.bus import EventBus

log = logging.getLogger(__name__)

StreamFactory = Callable[[list[str]], AsyncIterator[MarketEvent]]
Unsubscribe = Callable[[list[str]], Awaitable[None]]


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
        self._symbols = list(dict.fromkeys(symbols))
        self._version = 0
        self._changed = asyncio.Condition()
        self._bus = bus
        self._book_depth = book_depth
        self._max_backoff_s = max_backoff_s

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    async def set_symbols(self, symbols: Sequence[str]) -> None:
        new = list(dict.fromkeys(symbols))
        if new == self._symbols:
            return
        async with self._changed:
            self._symbols = new
            self._version += 1
            self._changed.notify_all()

    async def run(self) -> None:
        try:
            async with asyncio.TaskGroup() as tg:
                for src in self._sources:
                    tg.create_task(
                        self._supervise(
                            f"{src.exchange}:trades",
                            src.stream_trades,
                            src.unsubscribe_trades,
                        )
                    )
                    tg.create_task(
                        self._supervise(
                            f"{src.exchange}:books",
                            lambda symbols, src=src: src.stream_order_books(
                                symbols, self._book_depth
                            ),
                            src.unsubscribe_order_books,
                        )
                    )
        finally:
            await asyncio.gather(*(src.close() for src in self._sources), return_exceptions=True)

    async def _supervise(
        self, name: str, open_stream: StreamFactory, unsubscribe: Unsubscribe
    ) -> None:
        """Keep a stream alive: reconnect with backoff on errors, restart on symbol changes."""
        backoff = 1.0
        subscribed: list[str] = []
        while True:
            version, symbols = self._version, list(self._symbols)
            if removed := [s for s in subscribed if s not in symbols]:
                try:
                    await unsubscribe(removed)
                except Exception:
                    log.warning("stream %s: unsubscribe %s failed", name, removed, exc_info=True)
            subscribed = symbols
            if not symbols:
                await self._wait_for_change(version)
                continue

            events = 0

            async def pump_events(stream: AsyncIterator[MarketEvent]) -> None:
                nonlocal events
                async for event in stream:
                    self._bus.publish(event)
                    events += 1

            pump = asyncio.create_task(pump_events(open_stream(symbols)))
            change = asyncio.create_task(self._wait_for_change(version))
            try:
                await asyncio.wait({pump, change}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                pump.cancel()
                change.cancel()
                # Wait for both to settle; pump's own error is inspected below, not raised here.
                await asyncio.gather(pump, change, return_exceptions=True)

            if change.done() and not change.cancelled():
                log.info("stream %s: symbols changed to %s", name, self._symbols)
                backoff = 1.0
                continue
            if (error := pump.exception()) is None:
                log.warning("stream %s ended, restarting", name)
                continue
            if events:  # the stream was healthy before failing: reconnect quickly
                backoff = 1.0
            log.error("stream %s failed, retrying in %.0fs", name, backoff, exc_info=error)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff_s)

    async def _wait_for_change(self, version: int) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: self._version != version)
