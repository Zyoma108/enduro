from types import SimpleNamespace

from test_agent import ScriptedLLM, runtime

from enduro.agent.runtime import AgentConfig
from enduro.analytics.impulse import detect_impulse, market_context
from enduro.core.models import Candle

T0 = 1_790_726_400_000


def bar(minute: int, close: float, volume: float = 1.0, symbol: str = "SOL/USDT:USDT") -> Candle:
    return Candle(
        "binance", symbol, T0 + minute * 60_000, close, close + 0.1, close - 0.1, close, volume
    )


def quiet(minutes: int = 300) -> list[Candle]:
    return [bar(i, 100.0) for i in range(minutes)]


def test_break_of_the_range_on_heavy_volume_is_an_impulse():
    candles = quiet() + [bar(300 + i, 100.5 + i * 0.3, volume=30.0) for i in range(5)]
    impulse = detect_impulse("SOL/USDT:USDT", candles, min_volume_vs_24h=10)
    assert impulse is not None and impulse.direction == "up"
    assert impulse.volume_vs_24h > 10 and impulse.beyond_range_pct > 0
    assert "impulse up on SOL/USDT:USDT" in impulse.describe()


def test_no_impulse_on_light_volume_or_inside_the_range():
    light = quiet() + [bar(300 + i, 100.5 + i * 0.3, volume=2.0) for i in range(5)]
    assert detect_impulse("SOL/USDT:USDT", light, 10) is None
    inside = quiet() + [bar(300 + i, 100.0, volume=30.0) for i in range(5)]
    assert detect_impulse("SOL/USDT:USDT", inside, 10) is None
    assert detect_impulse("SOL/USDT:USDT", quiet(50), 10) is None


def test_market_context_has_leaders_and_breadth():
    up = [bar(i, 100 + i * 0.01) for i in range(300)]
    down = [bar(i, 100 - i * 0.01, symbol="X") for i in range(300)]
    view = market_context({"BTC/USDT:USDT": up}, {"BTC": up, "X": down, "Y": up})
    assert view["BTC"]["1h"] > 0 and "24h" not in view["BTC"]  # not enough history for 24h
    assert view["universe"]["1h"]["share_up"] == 0.67


async def test_impulse_wakes_a_flat_agent_once_and_is_listed(tmp_path, monkeypatch):
    monkeypatch.setattr("enduro.agent.runtime.now_ms", lambda: T0 + 306 * 60_000)
    rt, *_ = runtime(tmp_path, ScriptedLLM([]), AgentConfig(impulse_volume_vs_24h=10))
    rt.radar.radar.by_symbol["SOL/USDT:USDT"] = quiet() + [
        bar(300 + i, 100.5 + i * 0.3, volume=30.0) for i in range(5)
    ]
    rt._check_alerts()
    assert rt._wake.is_set() and "impulse up on SOL/USDT:USDT" in rt._wake_reason
    rt._wake.clear()
    rt._check_alerts()  # same impulse: cooldown
    assert not rt._wake.is_set() and len(rt._impulses) == 1


async def test_trailing_stop_follows_the_best_price_only_in_profit(tmp_path):
    rt, trading, *_ = runtime(tmp_path, ScriptedLLM([]))
    position = SimpleNamespace(
        symbol="SOL/USDT:USDT", side="long", entry_price=100.0, stop_loss=98.0, mark_price=100.0
    )

    class Gateway:
        async def positions(self, symbols=None):
            return [position]

    moves = []

    async def protect(symbol, side, stop, take):
        moves.append(stop)
        position.stop_loss = stop
        return {"updated": True}

    trading.gateway, trading.protect = Gateway(), protect
    rt.set_trail("SOL/USDT:USDT", "long", 0.01)
    for price in (100.5, 101.5, 101.2, 103.0, 103.05):
        position.mark_price = price
        await rt._trail_once()
    # 100.5: stop would sit below entry; 101.5 -> ~100.48; 101.2: best unchanged;
    # 103.0 -> 101.97; 103.05: too small a step
    assert [round(m, 2) for m in moves] == [100.48, 101.97]


async def test_trail_of_a_closed_position_is_dropped_at_the_next_tick(tmp_path):
    rt, *_ = runtime(tmp_path, ScriptedLLM([]))
    rt.set_trail("SOL/USDT:USDT", "long", 0.01)
    await rt._situation("search", "scheduled")
    assert rt.trails == {}


async def test_a_stop_inside_the_5m_noise_is_refused(tmp_path):
    from test_agent import FakeRow, call, turn

    llm = ScriptedLLM(
        [
            [
                turn(call("set_focus", symbol="ETH/USDT:USDT", reason="x")),
                turn(call("open_position", side="long", stop_loss=99.0, thesis="t")),
                turn(call("open_position", side="long", stop_loss=98.0, thesis="t")),
                turn(call("finish_tick", next_check_seconds=60, note="n")),
            ]
        ]
    )
    rt, trading, *_ = runtime(tmp_path, llm)
    rt.radar.rows = [FakeRow("ETH/USDT:USDT", atr_5m=0.01)]
    rt.radar.rows[0].price = 100.0
    await rt.tick("start")
    results = [r[0] for kind, r in llm.log if kind == "results"]
    assert results[1].is_error and "1.00 ATR(5m)" in results[1].content
    assert not results[2].is_error and len(trading.opened) == 1
