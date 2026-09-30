import asyncio

from enduro.core.models import OrderBook, Trade
from enduro.storage import store
from enduro.storage.parquet_sink import ParquetSink

DAY1 = 1_790_726_400_000  # 2026-09-30 00:00:00 UTC
DAY2 = DAY1 + 86_400_000


def trade(ts: int, exchange: str = "binance", symbol: str = "BTC/USDT:USDT") -> Trade:
    return Trade(exchange, symbol, ts, ts + 5, price=100.0, amount=0.5, side="buy", id=str(ts))


def book(ts: int, exchange: str = "bybit") -> OrderBook:
    return OrderBook(
        exchange, "BTC/USDT:USDT", ts, ts, bids=((99.0, 1.0), (98.0, 2.0)), asks=((101.0, 3.0),)
    )


async def test_round_trip_and_partitioning(tmp_path):
    sink = ParquetSink(tmp_path, book_interval_ms=1_000)
    for event in [
        trade(DAY1 + 1),
        trade(DAY1 + 2, symbol="ETH/USDT:USDT"),
        trade(DAY2 + 1, exchange="bybit"),
        book(DAY1),
        book(DAY1 + 500),  # dropped: within the 1s interval
        book(DAY1 + 1_000),
    ]:
        sink.add(event)
    await sink.flush()

    assert sink.rows_written == 5
    assert len(list((tmp_path / "trades" / "date=2026-09-30" / "exchange=binance").iterdir())) == 1
    assert (tmp_path / "trades" / "date=2026-10-01" / "exchange=bybit").is_dir()

    con = store.connect(tmp_path)
    rows = con.sql(
        "select exchange, symbol, price, side, id, recv_ts - ts, date::varchar "
        "from trades order by ts"
    ).fetchall()
    assert rows == [
        ("binance", "BTC/USDT:USDT", 100.0, "buy", str(DAY1 + 1), 5, "2026-09-30"),
        ("binance", "ETH/USDT:USDT", 100.0, "buy", str(DAY1 + 2), 5, "2026-09-30"),
        ("bybit", "BTC/USDT:USDT", 100.0, "buy", str(DAY2 + 1), 5, "2026-10-01"),
    ]
    books = con.sql("select ts, bid_px, bid_sz, ask_px from books order by ts").fetchall()
    assert books == [
        (DAY1, [99.0, 98.0], [1.0, 2.0], [101.0]),
        (DAY1 + 1_000, [99.0, 98.0], [1.0, 2.0], [101.0]),
    ]


async def test_book_downsampling_is_per_exchange_and_can_be_disabled(tmp_path):
    sink = ParquetSink(tmp_path, book_interval_ms=1_000)
    sink.add(book(DAY1, exchange="bybit"))
    sink.add(book(DAY1 + 1, exchange="binance"))  # different exchange: kept
    sink.add(book(DAY1 + 2, exchange="bybit"))  # same exchange within interval: dropped
    assert sum(len(rows) for rows in sink._buffer.values()) == 2

    keep_all = ParquetSink(tmp_path, book_interval_ms=0)
    for ts in range(3):
        keep_all.add(book(DAY1 + ts))
    assert sum(len(rows) for rows in keep_all._buffer.values()) == 3


async def test_run_flushes_remaining_events_on_cancel(tmp_path):
    sink = ParquetSink(tmp_path, flush_interval_s=3600)
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(sink.run(queue))
    queue.put_nowait(trade(DAY1))
    await asyncio.sleep(0.01)
    queue.put_nowait(trade(DAY1 + 1))  # still queued at cancellation time
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert sink.rows_written == 2
    assert store.connect(tmp_path).sql("select count(*) from trades").fetchone() == (2,)


def test_connect_on_empty_root(tmp_path):
    con = store.connect(tmp_path)
    assert con.sql("select count(*) from duckdb_views() where not internal").fetchone() == (0,)
