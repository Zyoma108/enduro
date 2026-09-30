"""In-process fan-out of market events to independent consumers."""

from __future__ import annotations

import asyncio

from enduro.core.models import MarketEvent


class EventBus:
    """Publishes every event to all subscriber queues.

    Publishing never blocks: a slow consumer must not stall data collection. When a
    subscriber's queue is full its oldest event is dropped (stale market data is
    worth less than fresh data) and counted in `dropped`.
    """

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue[MarketEvent]] = []
        self.dropped = 0

    def subscribe(self, maxsize: int = 10_000) -> asyncio.Queue[MarketEvent]:
        queue: asyncio.Queue[MarketEvent] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    def publish(self, event: MarketEvent) -> None:
        for queue in self._queues:
            if queue.full():
                queue.get_nowait()
                self.dropped += 1
            queue.put_nowait(event)
