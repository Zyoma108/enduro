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

    async def unsubscribe_trades(self, symbols):
        pass

    async def unsubscribe_order_books(self, symbols):
        pass

    async def reset_streams(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


class RecordingSource:
    """Streams one trade per symbol, then idles; records subscriptions."""

    exchange = "rec"

    def __init__(self) -> None:
        self.opened: list[tuple[str, ...]] = []
        self.unsubscribed: list[tuple[str, ...]] = []

    async def stream_trades(self, symbols):
        self.opened.append(tuple(symbols))
        for s in symbols:
            yield Trade("rec", s, 0, 0, price=1.0, amount=1.0, side="buy")
        await asyncio.Event().wait()

    async def stream_order_books(self, symbols, depth):
        await asyncio.Event().wait()
        yield  # pragma: no cover

    async def unsubscribe_trades(self, symbols):
        self.unsubscribed.append(tuple(symbols))

    async def unsubscribe_order_books(self, symbols):
        pass

    async def reset_streams(self) -> None:
        pass

    async def close(self) -> None:
        pass


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


async def test_collector_switches_symbols_at_runtime():
    bus = EventBus()
    queue = bus.subscribe()
    source = RecordingSource()
    collector = Collector([source], [], bus)
    task = asyncio.create_task(collector.run())

    async def next_symbols(n: int) -> list[str]:
        return [(await asyncio.wait_for(queue.get(), timeout=1)).symbol for _ in range(n)]

    await asyncio.sleep(0.01)
    assert source.opened == []  # nothing to stream yet

    await collector.set_symbols(["A", "B"])
    assert await next_symbols(2) == ["A", "B"]

    await collector.set_symbols(["B", "C"])
    assert await next_symbols(2) == ["B", "C"]
    await collector.set_symbols(["B", "C"])  # no-op: same set

    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert source.opened == [("A", "B"), ("B", "C")]
    assert source.unsubscribed == [("A",)]
    assert collector.symbols == ["B", "C"]


class ResubscribeSource(RecordingSource):
    """The first trade subscription fails as if the server still had a stale one."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_once = True

    async def stream_trades(self, symbols):
        self.opened.append(tuple(symbols))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("already subscribed")
        for s in symbols:
            yield Trade("rec", s, 0, 0, price=1.0, amount=1.0, side="buy")
        await asyncio.Event().wait()


async def test_failed_stream_resets_its_subscription_before_retrying(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _s: real_sleep(0))
    bus = EventBus()
    queue = bus.subscribe()
    source = ResubscribeSource()
    task = asyncio.create_task(Collector([source], ["A"], bus).run())
    event = await asyncio.wait_for(queue.get(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert event.symbol == "A"
    assert source.opened == [("A",), ("A",)]
    assert source.unsubscribed == [("A",)]  # reset between the failure and the retry


class SilentBookSource(RecordingSource):
    """Book stream sends one snapshot and then goes silent, like a dead socket; trades
    are cancelled under the reader when the connection is reset (as ccxt does)."""

    exchange = "silent"

    def __init__(self) -> None:
        super().__init__()
        self.book_opens = 0
        self.trade_opens = 0
        self.resets = 0
        self._pending: asyncio.Future | None = None

    async def stream_order_books(self, symbols, depth):
        self.book_opens += 1
        yield OrderBook("silent", symbols[0], 0, 0, bids=((1.0, 1.0),), asks=((2.0, 1.0),))
        await asyncio.Event().wait()

    async def stream_trades(self, symbols):
        self.trade_opens += 1
        self._pending = asyncio.get_running_loop().create_future()
        await self._pending  # ccxt-style pending read: cancelled by a reset
        yield  # pragma: no cover

    async def reset_streams(self) -> None:
        self.resets += 1
        if self._pending and not self._pending.done():
            self._pending.cancel()


async def test_silent_book_stream_is_reset_and_both_streams_reopen():
    source = SilentBookSource()
    collector = Collector([source], ["X"], EventBus(), book_stall_s=0.05)
    task = asyncio.create_task(collector.run())
    for _ in range(200):
        await asyncio.sleep(0.01)
        if source.book_opens >= 2 and source.trade_opens >= 2:
            break
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert source.resets >= 1
    assert source.book_opens >= 2  # the silent book stream was reopened
    assert source.trade_opens >= 2  # the sibling stream survived the reset and reopened


async def test_recent_trades_skips_a_failing_source():
    class History(RecordingSource):
        async def fetch_recent_trades(self, symbol, since):
            return [Trade("rec", symbol, since + 1, since + 1, price=1.0, amount=1.0, side="buy")]

    class Broken(RecordingSource):
        exchange = "broken"

        async def fetch_recent_trades(self, symbol, since):
            raise RuntimeError("down")

    collector = Collector([History(), Broken()], [], EventBus())
    got = await collector.recent_trades("X", 10)
    assert list(got) == ["rec"] and got["rec"][0].ts == 11
