import asyncio

from enduro.core.models import OrderBook, Trade
from enduro.data.bus import EventBus
from enduro.data.collector import Collector


def make_trade(ts: int = 0, exchange: str = "fake") -> Trade:
    return Trade(exchange, "X", ts, ts, price=1.0, amount=1.0, side="buy")


class FlakySource:
    """Emits one trade, then fails; emits one book per call."""

    exchange = "fake"

    def __init__(self) -> None:
        self.trade_calls = 0
        self.closed = False

    async def stream_trades(self, symbols):
        self.trade_calls += 1
        yield make_trade(self.trade_calls)
        raise ConnectionError("boom")

    async def stream_order_books(self, symbols, depth):
        yield OrderBook("fake", "X", 0, 0, bids=((1.0, 1.0),), asks=((2.0, 1.0),))
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed = True


def test_bus_fans_out_and_drops_oldest():
    bus = EventBus()
    small, big = bus.subscribe(maxsize=2), bus.subscribe()
    for ts in range(3):
        bus.publish(make_trade(ts))
    assert [small.get_nowait().ts for _ in range(2)] == [1, 2]
    assert big.qsize() == 3
    assert bus.dropped == 1


async def test_collector_reconnects_and_closes_sources(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _s: real_sleep(0))

    bus = EventBus()
    queue = bus.subscribe()
    source = FlakySource()
    task = asyncio.create_task(Collector([source], ["X"], bus).run())

    trades: list[Trade] = []
    books = 0
    while len(trades) < 3:
        event = await asyncio.wait_for(queue.get(), timeout=1)
        if isinstance(event, Trade):
            trades.append(event)
        else:
            books += 1

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert [t.ts for t in trades] == [1, 2, 3]
    assert books == 1
    assert source.closed
