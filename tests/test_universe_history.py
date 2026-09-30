from enduro.core.models import MINUTE_MS, Candle
from enduro.data.ccxt_source import asset_class, candle_from_ccxt, is_transient
from enduro.data.history import backfill, fetch_closed_candles
from enduro.data.universe import build_universe, common_symbols, select_universe
from enduro.storage import store
from enduro.storage.candles import last_candle_ts, write_candles

T0 = 1_790_726_400_000  # minute-aligned


def candle(ts: int, symbol: str = "X", exchange: str = "fake") -> Candle:
    return Candle(exchange, symbol, ts, 1.0, 2.0, 0.5, 1.5, 10.0)


class FakeSource:
    """Serves 1m candles from `first` up to (and including the still-open) `now` minute."""

    exchange = "fake"

    def __init__(self, first: int, now: int, markets=None, volumes=None, page: int = 3):
        self.first, self.now, self.page = first, now, page
        self.markets, self.volumes = markets or {}, volumes or {}
        self.requests = 0

    async def fetch_candles(self, symbol, since, limit=1000, timeframe="1m"):
        self.requests += 1
        start = max(since, self.first)
        last_open = self.now // MINUTE_MS * MINUTE_MS
        stamps = range(start, last_open + 1, MINUTE_MS)
        return [candle(ts, symbol) for ts in list(stamps)[: min(limit, self.page)]]

    async def list_linear_usdt_perps(self):
        return self.markets

    async def fetch_quote_volumes(self, symbols):
        return {s: self.volumes[s] for s in symbols if s in self.volumes}


async def test_fetch_closed_candles_keeps_pages_fetched_before_an_error():
    now = T0 + 10 * MINUTE_MS
    source = FakeSource(first=T0, now=now, page=3)
    real_fetch = source.fetch_candles

    async def flaky(symbol, since, limit=1000, timeframe="1m"):
        if since >= T0 + 6 * MINUTE_MS:
            raise RuntimeError("svc error")
        return await real_fetch(symbol, since, limit, timeframe)

    source.fetch_candles = flaky
    candles = await fetch_closed_candles(source, "X", since=T0, now_ms=now)
    assert [c.ts for c in candles] == [T0 + i * MINUTE_MS for i in range(6)]


def test_is_transient():
    import ccxt

    assert is_transient(ccxt.RequestTimeout("timeout"))
    assert is_transient(ccxt.ExchangeError('bybit {"retCode":10016,"retMsg":"svc error"}'))
    assert not is_transient(ccxt.ExchangeError('bybit {"retCode":10001,"retMsg":"bad param"}'))
    assert not is_transient(ValueError("x"))


def test_asset_class():
    assert asset_class("bybit", {"info": {"symbolType": ""}}) == "crypto"
    assert asset_class("bybit", {"info": {"symbolType": "innovation"}}) == "crypto"
    assert asset_class("bybit", {"info": {"symbolType": "stock"}}) == "tradfi"
    assert asset_class("binance", {"info": {"underlyingType": "COIN"}}) == "crypto"
    assert asset_class("binance", {"info": {"underlyingType": "COMMODITY"}}) == "tradfi"
    assert asset_class("okx", {"info": {}}) == "crypto"


def test_candle_from_ccxt():
    c = candle_from_ccxt("bybit", "X", [T0, "1", "2", "0.5", "1.5", None])
    assert (c.ts, c.open, c.high, c.low, c.close, c.volume) == (T0, 1.0, 2.0, 0.5, 1.5, 0.0)


def test_common_symbols_requires_allowed_class_on_both():
    reference = {"BTC": "crypto", "NVDA": "tradfi", "XAUT": "crypto", "ONLYREF": "crypto"}
    execution = {"BTC": "crypto", "NVDA": "tradfi", "XAUT": "tradfi", "ONLYEXEC": "crypto"}
    assert common_symbols(reference, execution, ["crypto"]) == ["BTC"]
    assert common_symbols(reference, execution, ["crypto", "tradfi"]) == ["BTC", "NVDA", "XAUT"]


def test_select_universe_filters_and_sorts():
    volumes = {"A": 5.0, "B": 20.0, "C": 50.0}
    entries = select_universe(["A", "B", "C", "D"], volumes, min_quote_volume=10.0)
    assert [(e.symbol, e.quote_volume_24h) for e in entries] == [("C", 50.0), ("B", 20.0)]


async def test_build_universe():
    ref = FakeSource(0, 0, markets={"A": "crypto", "B": "crypto"})
    exe = FakeSource(0, 0, markets={"A": "crypto", "B": "crypto"}, volumes={"A": 1, "B": 99})
    entries = await build_universe(ref, exe, min_quote_volume=10)
    assert [e.symbol for e in entries] == ["B"]


async def test_fetch_closed_candles_pages_and_skips_open_candle():
    now = T0 + 7 * MINUTE_MS + 30_000  # the T0+7m candle is still open
    source = FakeSource(first=T0, now=now, page=3)
    candles = await fetch_closed_candles(source, "X", since=T0 + 5, now_ms=now)
    # `since` is floored to its minute; the open T0+7m candle is dropped
    assert [c.ts for c in candles] == [T0 + i * MINUTE_MS for i in range(7)]
    assert source.requests == 3


async def test_backfill_resumes_after_last_stored_candle():
    now = T0 + 10 * MINUTE_MS
    source = FakeSource(first=T0 - 100 * MINUTE_MS, now=now, page=1000)
    got: dict[str, list[int]] = {}
    total = await backfill(
        source,
        ["NEW", "OLD", "UPTODATE"],
        last_ts={"OLD": T0 + 7 * MINUTE_MS, "UPTODATE": T0 + 9 * MINUTE_MS},
        start_ms=T0,
        now_ms=now,
        on_candles=lambda s, cs: got.__setitem__(s, [c.ts for c in cs]),
    )
    assert got["NEW"] == [T0 + i * MINUTE_MS for i in range(10)]
    assert got["OLD"] == [T0 + 8 * MINUTE_MS, T0 + 9 * MINUTE_MS]
    assert "UPTODATE" not in got
    assert total == 12


def test_candle_storage_round_trip(tmp_path):
    assert last_candle_ts(tmp_path, "binance") == {}
    write_candles(tmp_path, [candle(T0, "A", "binance"), candle(T0 + MINUTE_MS, "A", "binance")])
    write_candles(tmp_path, [candle(T0 + 2 * MINUTE_MS, "A", "binance"), candle(T0, "B", "bybit")])

    assert last_candle_ts(tmp_path, "binance") == {"A": T0 + 2 * MINUTE_MS}
    assert last_candle_ts(tmp_path, "bybit") == {"B": T0}
    row = (
        store.connect(tmp_path)
        .sql("select count(*), min(timeframe), max(high) from candles where exchange = 'binance'")
        .fetchone()
    )
    assert row == (3, "1m", 2.0)
    assert not list(tmp_path.glob("**/*.tmp"))
