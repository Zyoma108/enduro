import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from enduro.agent.llm import LLMTurn, ToolCall, ToolResult, Usage, UsageLimitError
from enduro.agent.prompt import render_prompt
from enduro.agent.runtime import AgentConfig, AgentRuntime, wake_threshold_bps
from enduro.agent.tools import TOOLS
from enduro.analytics.focus import FocusTracker
from enduro.analytics.liquidity import Liquidity, LiquidityBook, liquidity_from_book
from enduro.core.models import Candle, OrderBook, Trade, now_ms
from enduro.execution.models import Position
from enduro.journal.journal import Journal
from enduro.risk.manager import RiskLimits, RiskManager, RiskStateStore

FEE = 0.00055


# ---------------------------------------------------------------- fakes


class ScriptedSession:
    """Plays back a list of turns; records what the agent sent."""

    def __init__(self, turns: list[LLMTurn], log: list) -> None:
        self.turns, self.log = turns, log

    async def send(self, text: str) -> LLMTurn:
        self.log.append(("user", text))
        return self.turns.pop(0)

    async def send_tool_results(self, results: list[ToolResult], text: str = "") -> LLMTurn:
        self.log.append(("results", results))
        return self.turns.pop(0)


class ScriptedLLM:
    name = "scripted"

    def __init__(self, episodes: list[list[LLMTurn]]) -> None:
        self.episodes = episodes
        self.log: list = []
        self.sessions = 0

    def session(self, system, tools):
        self.sessions += 1
        return ScriptedSession(self.episodes.pop(0), self.log)


def call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"c-{name}", name=name, input=args)


def turn(*calls: ToolCall, text: str = "") -> LLMTurn:
    return LLMTurn(text, list(calls), "tool_use" if calls else "end_turn", Usage(cost_usd=0.01))


@dataclass
class FakeRow:
    symbol: str
    atr_5m: float = 0.01
    score: float = 1.0

    def to_summary(self):
        return {"symbol": self.symbol, "score": self.score}


class FakeCandles:
    """Stands in for the Radar's candle buffer."""

    def __init__(self) -> None:
        self.by_symbol: dict[str, list[Candle]] = {}

    def candles(self, symbol, minutes):
        return self.by_symbol.get(symbol, [])[-minutes:]


class FakeRadar:
    updated_ms = 1

    def __init__(self) -> None:
        self.rows: list = []
        self.radar = FakeCandles()
        self.liquidity = LiquidityBook(min_depth_usd=2_000, max_spread_bps=10)

    async def refresh(self):
        return []


class FakeCollector:
    def __init__(self) -> None:
        self.symbols: list[str] = []
        self.history: dict[str, list[Trade]] = {}  # exchange -> trades REST would return
        self.history_requests: list[str] = []

    async def set_symbols(self, symbols):
        self.symbols = list(symbols)

    async def recent_trades(self, symbol, since, timeout_s=None):
        self.history_requests.append(symbol)
        return {
            ex: [t for t in trades if t.symbol == symbol] for ex, trades in self.history.items()
        }


class FakeTrading:
    def __init__(self) -> None:
        self.opened: list = []
        self.positions: list[dict] = []

    async def account(self):
        return {"equity_usdt": 1000.0, "positions": self.positions, "risk": {}}

    async def open(self, intent, thesis):
        self.opened.append((intent, thesis))
        return {"opened": True, "qty": 1.0, "avg_price": 120.0}


def runtime(
    tmp_path, llm, config: AgentConfig | None = None
) -> tuple[AgentRuntime, FakeTrading, FakeCollector, Journal]:
    trading, collector = FakeTrading(), FakeCollector()
    journal = Journal(tmp_path / "journal")
    rt = AgentRuntime(
        llm=llm,
        system_prompt="spirit",
        trading=trading,
        radar=FakeRadar(),
        focus_collector=collector,
        focus_tracker=FocusTracker("binance", "bybit"),
        reference_source=None,
        journal=journal,
        config=config or AgentConfig(),
        universe=["SOL/USDT:USDT", "ETH/USDT:USDT"],
        reference="binance",
        execution="bybit",
    )
    return rt, trading, collector, journal


# ---------------------------------------------------------------- agent loop


async def test_search_then_focus_then_trade(tmp_path):
    llm = ScriptedLLM(
        [
            [  # tick 1: search
                turn(call("get_radar", top=5)),
                turn(call("set_focus", symbol="SOL/USDT:USDT", reason="hot and trending")),
                turn(call("finish_tick", next_check_seconds=30, note="watching SOL breakout")),
            ],
            [  # tick 2: focus — first a bad call, then a valid entry
                turn(call("open_position", side="long", thesis="buyers in control")),  # no stop
                turn(call("open_position", side="sideways", stop_loss=99, thesis="x")),
                turn(
                    call("open_position", side="long", stop_loss=118.5, thesis="buyers in control")
                ),
                turn(call("finish_tick", next_check_seconds=20, note="long SOL, stop 118.5")),
            ],
        ]
    )
    rt, trading, collector, journal = runtime(tmp_path, llm)

    await rt.tick("start")
    assert rt.focus_symbol == "SOL/USDT:USDT"
    assert collector.symbols == ["SOL/USDT:USDT"]
    first_user = llm.log[0][1]
    assert "mode: search" in first_user and "## Radar" in first_user

    await rt.tick("scheduled")
    assert "mode: focus" in [m for kind, m in llm.log if kind == "user"][1]
    errors = [r[0] for kind, r in llm.log if kind == "results" and r[0].is_error]
    assert "'stop_loss' is required" in errors[0].content
    assert "'side' must be" in errors[1].content
    intent, thesis = trading.opened[0]
    assert (intent.symbol, intent.side, intent.stop_loss) == ("SOL/USDT:USDT", "long", 118.5)
    assert thesis == "buyers in control"

    notes = journal.recent("note", 10)
    assert [n["text"] for n in notes] == ["watching SOL breakout", "long SOL, stop 118.5"]
    kinds = [r["kind"] for r in journal.read()]
    assert kinds.count("tick") == 2 and kinds.count("llm") == 7
    assert rt.session_cost_usd == pytest.approx(0.07)


async def test_notes_are_carried_into_the_next_tick(tmp_path):
    llm = ScriptedLLM(
        [
            [turn(call("finish_tick", next_check_seconds=120, note="nothing clean yet"))],
            [turn(text="still nothing")],  # ends without finish_tick: text becomes the note
        ]
    )
    rt, *_, journal = runtime(tmp_path, llm)
    await rt.tick("start")
    await rt.tick("scheduled")
    second_user = [m for kind, m in llm.log if kind == "user"][1]
    assert "nothing clean yet" in second_user
    assert journal.recent("note", 5)[-1]["text"] == "still nothing"


async def test_unknown_symbol_and_trading_without_focus_are_errors(tmp_path):
    llm = ScriptedLLM(
        [
            [
                turn(call("set_focus", symbol="DOGE/USDT:USDT", reason="meme")),
                turn(call("open_position", side="long", stop_loss=1, thesis="x")),
                turn(call("finish_tick", next_check_seconds=60, note="n")),
            ]
        ]
    )
    rt, trading, *_ = runtime(tmp_path, llm)
    await rt.tick("start")
    results = [r[0] for kind, r in llm.log if kind == "results"]
    assert results[0].is_error and "not in the scanned universe" in results[0].content
    assert results[1].is_error and "set_focus first" in results[1].content
    assert trading.opened == []


async def test_tick_stops_after_max_llm_calls(tmp_path):
    llm = ScriptedLLM([[turn(call("get_account")) for _ in range(10)]])
    rt, *_, journal = runtime(tmp_path, llm, AgentConfig(max_llm_calls_per_tick=3))
    await rt.tick("start")
    assert sum(1 for r in journal.read() if r["kind"] == "llm") == 3
    assert any(r["kind"] == "error" for r in journal.read())


async def test_release_focus_requires_flat_position(tmp_path):
    llm = ScriptedLLM(
        [
            [
                turn(call("release_focus", reason="done")),
                turn(call("finish_tick", next_check_seconds=60, note="n")),
            ]
        ]
    )
    rt, trading, _, _ = runtime(tmp_path, llm)
    await rt.set_focus("SOL/USDT:USDT", "test")
    trading.positions = [{"symbol": "SOL/USDT:USDT", "side": "long"}]
    await rt.tick("start")
    result = next(r[0] for kind, r in llm.log if kind == "results")
    assert result.is_error and "close the open position" in result.content
    assert rt.focus_symbol == "SOL/USDT:USDT"


def test_tool_schemas_are_valid_and_strict():
    names = [t.spec.name for t in TOOLS]
    assert len(names) == len(set(names))
    for tool in TOOLS:
        schema = tool.spec.input_schema
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])
        json.dumps(schema)


# ---------------------------------------------------------------- prompt, journal, risk


def test_prompt_renders_live_limits():
    limits = RiskLimits(risk_per_trade_pct=0.5, daily_loss_limit_pct=3.0)
    text = render_prompt(Path("prompts/trader.md"), limits, 5.5, "demo")
    assert "до 0.5% капитала" in text and "дневной убыток 3%" in text
    assert "0.055%" in text and "демо-счёте" in text
    assert "$" not in text.replace("$risk", "")  # every placeholder substituted


def test_journal_appends_jsonl(tmp_path):
    journal = Journal(tmp_path)
    journal.write("note", text="привет")
    journal.write("tick", n=1)
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert [r["kind"] for r in journal.read()] == ["note", "tick"]
    assert journal.recent("note", 5)[0]["text"] == "привет"


def protection_manager(tmp_path) -> RiskManager:
    return RiskManager(RiskLimits(), FEE, RiskStateStore(tmp_path / "r.json"))


@pytest.mark.parametrize(
    ("stop", "take", "ok", "fragment"),
    [
        (99.5, None, True, ""),  # tighter than the current 99
        (101.0, None, True, ""),  # trailing into profit (below mark 102)
        (98.0, None, True, ""),  # looser, still within budget: 4 * (2 + fees) = ~8.4
        (0, None, False, "not allowed"),
        (102.5, None, False, "must be below"),
        (None, 101.0, False, "losing side"),
        (None, 0, True, ""),  # removing the take profit is fine
    ],
)
def test_evaluate_protection(tmp_path, stop, take, ok, fragment):
    rm = protection_manager(tmp_path)
    position = Position("X", "long", 4.0, 100.0, 102.0, 8.0, 5.0, None, stop_loss=99.0)
    allowed, reason = rm.evaluate_protection(
        position, stop_loss=stop, take_profit=take, mark=102.0, equity=1000.0
    )
    assert allowed is ok, reason
    assert fragment in reason


def test_loosening_a_stop_beyond_budget_is_rejected(tmp_path):
    rm = protection_manager(tmp_path)
    position = Position("X", "long", 4.0, 100.0, 102.0, 8.0, 5.0, None, stop_loss=99.0)
    allowed, reason = rm.evaluate_protection(
        position, stop_loss=97.0, take_profit=None, mark=102.0, equity=1000.0
    )
    assert not allowed and "budget" in reason  # 4 * (3 + fees) = 12.4 > 10


def test_scripted_llm_is_exhausted_cleanly():
    llm = ScriptedLLM([[turn(text="hi")]])
    session = llm.session("s", [])
    assert asyncio.run(session.send("x")).text == "hi"


async def test_tooling_feedback_is_journaled_and_shown_next_tick(tmp_path):
    llm = ScriptedLLM(
        [
            [
                turn(
                    call(
                        "report_tooling_gap",
                        category="missing_data",
                        title="no open interest",
                        details="cannot tell new longs from short covering",
                        impact="skipped a breakout",
                    ),
                    call("finish_tick", next_check_seconds=60, note="n"),
                )
            ],
            [
                turn(call("report_tooling_gap", category="wishes", title="x", details="y")),
                turn(call("finish_tick", next_check_seconds=60, note="n")),
            ],
        ]
    )
    rt, *_, journal = runtime(tmp_path, llm)
    await rt.tick("start")
    feedback = journal.recent("feedback", 5)
    assert feedback[0]["title"] == "no open interest"
    assert feedback[0]["impact"] == "skipped a breakout"

    await rt.tick("scheduled")
    second_user = [m for kind, m in llm.log if kind == "user"][1]
    assert "[missing_data] no open interest" in second_user
    bad = next(r[0] for kind, r in llm.log if kind == "results")
    assert bad.is_error and "'category' must be one of" in bad.content


# ---------------------------------------------------------------- wake-ups, liquidity, ATR


def test_wake_threshold_scales_with_volatility_and_position():
    cfg = AgentConfig(wake_move_bps=30, wake_move_atr=0.5, wake_flat_multiplier=2.0)
    assert wake_threshold_bps(0.035, True, cfg) == pytest.approx(175)  # MOVR-like, in position
    assert wake_threshold_bps(0.035, False, cfg) == pytest.approx(350)  # flat: doubled
    assert wake_threshold_bps(0.0015, True, cfg) == 30  # calm coin: the floor applies
    assert wake_threshold_bps(None, False, cfg) == 60


def liq(spread, bid, ask):
    return Liquidity(spread_bps=spread, depth_bid_usd=bid, depth_ask_usd=ask, ts=0)


def test_liquidity_from_book_and_median_judgement():
    book = OrderBook(
        "bybit", "X", 0, 0, bids=((99.95, 30.0), (99.0, 999.0)), asks=((100.05, 10.0),)
    )
    snapshot = liquidity_from_book(book)
    assert snapshot.spread_bps == pytest.approx(10.0)
    assert snapshot.depth_bid_usd == pytest.approx(99.95 * 30)  # 99.0 is outside 10 bps
    assert snapshot.depth_usd == pytest.approx(1000.5)  # the thinner side

    lb = LiquidityBook(min_depth_usd=2_000, max_spread_bps=10, keep=3)
    assert lb.tradable("X") is None  # unknown is not the same as illiquid
    lb.add({"X": liq(2, 5_000, 6_000)}, 1)
    lb.add({"X": liq(2, 500, 6_000)}, 2)  # one momentarily thin snapshot...
    lb.add({"X": liq(3, 4_000, 6_000)}, 3)
    assert lb.tradable("X") is True  # ...does not flip the median
    lb.add({"X": liq(15, 4_000, 6_000)}, 4)
    lb.add({"X": liq(14, 4_000, 6_000)}, 5)
    assert lb.tradable("X") is False  # wide spread for most of the window
    assert lb.summary("X")["bybit_spread_bps"] == 14


async def test_radar_view_hides_illiquid_coins_and_focus_shows_atr(tmp_path):
    rt, *_ = runtime(tmp_path, ScriptedLLM([]))
    rt.radar.rows = [FakeRow("SOL/USDT:USDT", atr_5m=0.012), FakeRow("ETH/USDT:USDT")]
    rt.radar.liquidity.add(
        {"SOL/USDT:USDT": liq(2, 50_000, 60_000), "ETH/USDT:USDT": liq(20, 100, 100)}, 1
    )
    view = rt.radar_view(10)
    assert [r["symbol"] for r in view["rows"]] == ["SOL/USDT:USDT"]
    assert view["hidden_illiquid_on_execution_exchange"] == 1
    assert view["rows"][0]["bybit_depth_10bps_usd"] == 50_000

    await rt.set_focus("SOL/USDT:USDT", "test")
    focus = await rt.focus_view()
    assert focus["atr_5m_pct"] == 1.2 and focus["tradable"] is True


async def test_open_reports_stop_distance_in_atr_and_focus_warns_on_illiquid(tmp_path):
    llm = ScriptedLLM(
        [
            [
                turn(call("set_focus", symbol="ETH/USDT:USDT", reason="x")),
                turn(call("open_position", side="long", stop_loss=119.76, thesis="t")),
                turn(call("finish_tick", next_check_seconds=60, note="n")),
            ]
        ]
    )
    rt, *_ = runtime(tmp_path, llm)
    rt.radar.rows = [FakeRow("ETH/USDT:USDT", atr_5m=0.01)]
    rt.radar.liquidity.add({"ETH/USDT:USDT": liq(20, 100, 100)}, 1)
    await rt.tick("start")
    results = [json.loads(r[0].content) for kind, r in llm.log if kind == "results"]
    assert "illiquid" in results[0]["warning"]
    assert results[1]["stop_distance_pct"] == pytest.approx(0.2)
    assert results[1]["stop_distance_in_atr_5m"] == pytest.approx(0.2)


async def test_flat_checks_are_spaced_out_but_positions_are_watched_closely(tmp_path):
    from enduro.core.models import now_ms

    llm = ScriptedLLM(
        [
            [turn(call("finish_tick", next_check_seconds=20, note="flat, wants 20 s"))],
            [turn(call("finish_tick", next_check_seconds=20, note="in position, 20 s"))],
        ]
    )
    rt, trading, *_ = runtime(tmp_path, llm, AgentConfig(min_check_flat_s=120))
    await rt.tick("start")
    assert (rt._next_check_ms - now_ms()) / 1000 == pytest.approx(120, abs=2)

    await rt.set_focus("SOL/USDT:USDT", "x")
    trading.positions = [{"symbol": "SOL/USDT:USDT", "side": "long"}]
    await rt.tick("scheduled")
    assert (rt._next_check_ms - now_ms()) / 1000 == pytest.approx(20, abs=2)


def test_prompt_states_the_flat_minimum():
    text = render_prompt(Path("prompts/trader.md"), RiskLimits(), 5.5, "demo", min_check_flat_s=150)
    assert "не раньше чем через 150 секунд" in text


# ---------------------------------------------------------------- price alerts


def candle(symbol: str, minute: int, close: float) -> Candle:
    ts = 1_790_726_400_000 + minute * 60_000
    return Candle("binance", symbol, ts, close, close, close, close, 1.0)


async def test_alert_is_set_listed_and_wakes_the_agent_once(tmp_path, monkeypatch):
    llm = ScriptedLLM(
        [
            [
                turn(
                    call(
                        "set_alert",
                        symbol="SOL/USDT:USDT",
                        level=99.5,
                        direction="below",
                        note="short the range break",
                    )
                ),
                turn(call("finish_tick", next_check_seconds=300, note="waiting for the break")),
            ],
        ]
    )
    rt, _, _, journal = runtime(tmp_path, llm)
    t0 = 1_790_726_400_000
    monkeypatch.setattr("enduro.agent.runtime.now_ms", lambda: t0 + 2 * 60_000)
    monkeypatch.setattr("enduro.agent.tools.now_ms", lambda: t0 + 2 * 60_000)
    rt.radar.radar.by_symbol["SOL/USDT:USDT"] = [candle("SOL/USDT:USDT", 1, 100.0)]
    await rt.tick("start")
    [alert] = rt.alerts.active(t0 + 2 * 60_000)
    assert (alert.level, alert.direction) == (99.5, "below")

    # the next closed candle is still above: nothing happens
    rt.radar.radar.by_symbol["SOL/USDT:USDT"].append(candle("SOL/USDT:USDT", 2, 99.8))
    rt._check_alerts()
    assert not rt._wake.is_set()

    # a close below the level fires it once and wakes the agent with the reason
    rt.radar.radar.by_symbol["SOL/USDT:USDT"].append(candle("SOL/USDT:USDT", 3, 99.2))
    rt._check_alerts()
    assert rt._wake.is_set()
    assert "alert #1 SOL/USDT:USDT: 1m candle " in rt._wake_reason
    assert "UTC closed 99.2 below 99.5" in rt._wake_reason
    assert rt.alerts.active(t0 + 4 * 60_000) == []
    records = journal.read(f"{datetime.now(UTC):%Y-%m-%d}")
    actions = [r["action"] for r in records if r["kind"] == "alert"]
    assert actions == ["set", "fired"]


async def test_alert_fired_during_a_tick_wakes_after_it(tmp_path, monkeypatch):
    rt, *_ = runtime(tmp_path, ScriptedLLM([]))
    t0 = 1_790_726_400_000
    monkeypatch.setattr("enduro.agent.runtime.now_ms", lambda: t0 + 2 * 60_000)
    rt.alerts.add("SOL/USDT:USDT", 99.5, "below", "x", 60, now_ms=t0)
    rt.radar.radar.by_symbol["SOL/USDT:USDT"] = [candle("SOL/USDT:USDT", 1, 99.0)]
    rt._in_tick = True
    rt._check_alerts()
    assert not rt._wake.is_set() and len(rt._fired_alerts) == 1
    rt._in_tick = False
    rt._wake_on_fired_alerts()
    assert rt._wake.is_set() and "alert #1" in rt._wake_reason


async def test_alert_that_would_fire_at_once_is_rejected(tmp_path):
    rt, *_ = runtime(tmp_path, ScriptedLLM([]))
    rt.radar.radar.by_symbol["SOL/USDT:USDT"] = [candle("SOL/USDT:USDT", 1, 100.0)]
    args = {"symbol": "SOL/USDT:USDT", "level": 101, "direction": "below", "note": "x"}
    result = await rt.call_tool("set_alert", args)
    assert result.is_error and "already below" in result.content
    result = await rt.call_tool("cancel_alert", {"id": 7})
    assert result.is_error and "no active alert #7" in result.content


# ---------------------------------------------------------------- session end


async def test_session_ends_right_after_the_last_tick(tmp_path):
    llm = ScriptedLLM([[turn(call("finish_tick", next_check_seconds=600, note="flat"))]])
    rt, *_ = runtime(tmp_path, llm)
    with pytest.raises(Exception) as info:  # _Done, not a 600 s wait
        await asyncio.wait_for(rt._ticks(1), 2)
    assert type(info.value).__name__ == "_Done"
    assert not rt.wind_down


async def test_tick_limit_with_open_position_keeps_managing_until_flat(tmp_path):
    llm = ScriptedLLM(
        [
            [turn(call("finish_tick", next_check_seconds=15, note="holding"))],
            [  # overtime: a new entry is refused, managing goes on
                turn(call("open_position", side="short", stop_loss=130, thesis="flip")),
                turn(call("finish_tick", next_check_seconds=15, note="closing soon")),
            ],
        ]
    )
    rt, trading, _, journal = runtime(tmp_path, llm)
    rt.focus_symbol = "SOL/USDT:USDT"
    trading.positions = [{"symbol": "SOL/USDT:USDT", "side": "long"}]
    original_tick = rt.tick

    async def tick(trigger):
        await original_tick(trigger)
        rt._next_check_ms = 0  # don't really wait between ticks
        if rt.tick_no == 2:
            trading.positions.clear()  # the agent closed it

    rt.tick = tick
    with pytest.raises(Exception) as info:
        await asyncio.wait_for(rt._ticks(1), 2)
    assert type(info.value).__name__ == "_Done"
    assert rt.tick_no == 2 and rt.wind_down
    assert trading.opened == []
    records = journal.read(f"{datetime.now(UTC):%Y-%m-%d}")
    refused = [r for r in records if r["kind"] == "tool" and r["name"] == "open_position"]
    assert "session is ending" in refused[0]["error"]
    assert any(r["kind"] == "session" for r in records)
    situation = next(text for kind, text in llm.log if kind == "user" and "Tick 2" in text)
    assert "## Session ending" in situation


# ---------------------------------------------------------------- closed trades


class FakeGateway:
    def __init__(self, trades) -> None:
        self.trades = trades

    async def closed_trades(self, limit=10):
        return self.trades[:limit]


async def test_exchange_closes_are_journaled_once_and_shown_to_the_agent(tmp_path):
    from enduro.core.models import now_ms
    from enduro.execution.models import ClosedTrade

    now = now_ms()
    aave, zro, qnt = "AAVE/USDT:USDT", "ZRO/USDT:USDT", "QNT/USDT:USDT"
    take_profit = ClosedTrade("tp-1", aave, "long", 5.06, 168.42, 170.4, 9.08, 0.94, now)
    ours = ClosedTrade("c-1", zro, "short", 211.9, 1.7465, 1.7545, -2.1, 0.41, now - 1)
    old = ClosedTrade("x-0", qnt, "short", 1.0, 1.0, 1.0, 0.0, 0.0, now - 2 * 86_400_000)
    llm = ScriptedLLM([[turn(call("finish_tick", next_check_seconds=60, note="n"))] for _ in "ab"])
    rt, trading, _, journal = runtime(tmp_path, llm)
    trading.gateway = FakeGateway([take_profit, ours, old])  # newest first
    journal.write("order", action="close", result={"id": "c-1"}, reason="thesis broken")
    # a previous session already journaled the ZRO close: it must not be written again
    journal.write("closed", trade=ours, closed_by="agent")

    await rt.tick("start")
    await rt.tick("scheduled")

    records = journal.read(f"{datetime.now(UTC):%Y-%m-%d}")
    closed = [r for r in records if r["kind"] == "closed"]
    assert [(r["trade"]["symbol"], r["closed_by"]) for r in closed] == [
        ("ZRO/USDT:USDT", "agent"),  # from the previous session
        ("AAVE/USDT:USDT", "exchange: stop loss / take profit"),
    ]  # once each; the 2-day-old close is not journaled late
    situation = next(text for kind, text in llm.log if kind == "user")
    section = situation.split("## Recently closed positions")[1]
    assert section.index("AAVE") < section.index("ZRO")  # newest first
    assert '"pnl_usdt_net_of_fees": 9.08' in section


async def test_closed_trades_carry_thesis_exit_reason_and_hindsight(tmp_path):
    from enduro.execution.models import ClosedTrade

    minute = 60_000
    now = now_ms() // minute * minute
    opened, closed = now - 10 * minute, now - 8 * minute
    trade = ClosedTrade("c-9", "SOL/USDT:USDT", "long", 1.0, 100.0, 99.5, -0.6, 0.1, closed)

    class Candles:
        calls = 0

        async def fetch_candles(self, symbol, since, limit=1000, timeframe="1m"):
            Candles.calls += 1
            return [
                Candle("bybit", symbol, opened, 100, 101, 99.8, 100.5, 1),
                Candle("bybit", symbol, opened + minute, 100.5, 100.6, 99.4, 99.5, 1),
                *[
                    Candle("bybit", symbol, closed + i * minute, 99, 99.2, 97.9, 98, 1)
                    for i in range(8)
                ],
            ]

    llm = ScriptedLLM([[turn(call("finish_tick", next_check_seconds=60, note="n"))] for _ in "ab"])
    rt, trading, _, journal = runtime(tmp_path, llm)
    rt.execution_source = Candles()
    trading.gateway = FakeGateway([trade])
    journal.write(
        "risk",
        intent={"symbol": trade.symbol, "side": "long"},
        decision={"approved": True},
        thesis="breakout of 99.9 with buyers",
    )
    journal.write(
        "order",
        action="open",
        result={"id": "o-9", "ts": opened},
        request={
            "symbol": trade.symbol,
            "position_side": "long",
            "stop_loss": 98.0,
            "take_profit": 103.0,
        },
    )
    # A chased close: our client tag, plus the exchange ids of every order it placed.
    journal.write(
        "order",
        action="close",
        result={"id": "enduro-1-c", "order_ids": ["c-8", "c-9"]},
        reason="back under the level",
    )

    await rt.tick("start")
    await rt.tick("scheduled")

    situation = next(text for kind, text in llm.log if kind == "user")
    line = next(x for x in situation.splitlines() if x.startswith('{"closed_utc"'))
    shown = json.loads(line)
    assert shown["your_thesis"] == "breakout of 99.9 with buyers"
    assert shown["your_exit_reason"] == "back under the level"
    assert (shown["stop_at_exit"], shown["take_profit_at_exit"]) == (98.0, 103.0)
    assert shown["hindsight"]["best_while_open_pct"] == 1.0
    assert shown["hindsight"]["if_held"] == "stop 98 would have been hit 1 min after your exit"
    assert Candles.calls == 1  # the second tick reused it (refreshed at most once a minute)


async def test_focus_shows_funding_from_both_exchanges_and_survives_one_failing(tmp_path):
    from enduro.core.models import Funding

    class Source:
        def __init__(self, exchange, rate, interval_h, fail=False):
            self.exchange, self.rate, self.interval_h, self.fail = exchange, rate, interval_h, fail
            self.calls = 0

        async def fetch_funding(self, symbol):
            self.calls += 1
            if self.fail:
                raise RuntimeError("down")
            next_ts = now_ms() + 90 * 60_000
            return Funding(self.exchange, symbol, now_ms(), self.rate, self.interval_h, next_ts)

    rt, _, _, _ = runtime(tmp_path, ScriptedLLM([]))
    rt.execution_source = bybit = Source("bybit", -0.0001, 8.0)
    rt.reference_source = Source("binance", 0.0003, 4.0, fail=True)
    await rt.set_focus("SOL/USDT:USDT", "test")
    funding = (await rt.focus_view())["funding"]
    assert funding["bybit"] == {
        "rate_pct": -0.01,
        "interval_h": 8.0,
        "per_day_pct": -0.03,
        "next_in_min": 90,
    }
    assert "binance" not in funding and "лонги платят" in funding["note"]
    await rt.focus_view()
    assert bybit.calls == 1  # cached for a minute


async def test_focus_shows_open_interest_next_to_price(tmp_path):
    from enduro.core.models import OpenInterest

    class Source:
        def __init__(self, exchange, fail=False):
            self.exchange, self.fail, self.calls = exchange, fail, 0

        async def fetch_funding(self, symbol):
            raise RuntimeError("not in this test")

        async def fetch_open_interest(self, symbol, minutes):
            self.calls += 1
            if self.fail:
                raise RuntimeError("down")
            now = now_ms() // 60_000 * 60_000
            return [
                OpenInterest(self.exchange, symbol, now - m * 60_000, 1000.0 - m)
                for m in (60, 15, 0)
            ]

    rt, _, _, _ = runtime(tmp_path, ScriptedLLM([]))
    rt.execution_source = Source("bybit", fail=True)
    rt.reference_source = binance = Source("binance")
    await rt.set_focus("SOL/USDT:USDT", "test")
    oi = (await rt.focus_view())["open_interest"]
    assert "bybit" not in oi and "в монетах" in oi["note"]
    assert oi["binance"]["15m"]["oi_pct"] == round((1000 / 985 - 1) * 100, 2)
    assert oi["binance"]["1h"]["oi_pct"] == round((1000 / 940 - 1) * 100, 2)
    await rt.focus_view()
    assert binance.calls == 1  # cached for a minute


# ---------------------------------------------------------------- resilience


async def test_tick_survives_an_exchange_outage_and_retries_soon(tmp_path):
    rt, trading, _, journal = runtime(tmp_path, ScriptedLLM([]))
    rt.focus_symbol = "SOL/USDT:USDT"
    trading.positions = [{"symbol": "SOL/USDT:USDT", "side": "long"}]
    rt._had_position = True

    async def down():
        raise ConnectionError("network is unreachable")

    trading.account = down
    await rt.tick("scheduled")  # must not raise
    errors = [r for r in journal.read(f"{datetime.now(UTC):%Y-%m-%d}") if r["kind"] == "error"]
    assert errors[0]["what"] == "tick" and "network is unreachable" in errors[0]["error"]
    assert rt._had_position  # unknown → keep the last known state
    from enduro.core.models import now_ms

    assert 25_000 <= rt._next_check_ms - now_ms() <= 31_000


async def test_radar_keeps_running_after_a_failed_refresh(monkeypatch):
    from enduro.analytics.radar_runner import RadarRunner

    runner = RadarRunner(None, [], Path("."), history_days=1, taker_fee_bps=5.5)
    calls = []

    async def refresh():
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("outage")
        return []

    async def no_sleep(_s):
        if len(calls) >= 2:
            raise asyncio.CancelledError

    runner.refresh = refresh
    monkeypatch.setattr("enduro.analytics.radar_runner.asyncio.sleep", no_sleep)
    updates = []
    with pytest.raises(asyncio.CancelledError):
        await runner.run_forever(updates.append)
    assert len(calls) == 2 and updates == [[]]  # the failure was survived, the retry updated


async def test_focus_is_backfilled_with_recent_trades(tmp_path):
    rt, _, collector, _ = runtime(tmp_path, ScriptedLLM([]))
    now = now_ms()
    collector.history = {
        "binance": [
            Trade("binance", "SOL/USDT:USDT", now - i * 1_000, now, 120, 1, "sell", id=str(i))
            for i in range(600, 0, -1)
        ],
        "bybit": [Trade("bybit", "SOL/USDT:USDT", now - 30_000, now, 120, 1, "buy", id="a")],
    }
    await rt.set_focus("SOL/USDT:USDT", "test")
    view = await rt.focus_view()
    assert view["flow"]["binance"]["5m"]["delta_ratio"] == -1.0
    assert "partial_window_s" not in view["flow"]["binance"]["5m"]
    assert view["observed_s"] == 30  # what both exchanges cover

    await rt.set_focus("SOL/USDT:USDT", "same coin again")
    assert collector.history_requests == ["SOL/USDT:USDT"]  # no second load


class LimitedLLM:
    """Every session fails with the account's usage limit."""

    name = "limited"

    def __init__(self, resets_at_ms: int | None) -> None:
        self.resets_at_ms = resets_at_ms

    def session(self, system, tools):
        resets_at_ms = self.resets_at_ms

        class Session:
            async def send(self, text):
                raise UsageLimitError("You've hit your session limit", resets_at_ms)

        return Session()


@pytest.mark.parametrize("known_reset", [True, False])
async def test_usage_limit_pauses_until_the_reset(tmp_path, known_reset):
    reset = now_ms() + 20 * 60_000
    rt, *_, journal = runtime(tmp_path, LimitedLLM(reset if known_reset else None))
    await rt.tick("start")
    expected = reset + 60_000 if known_reset else now_ms() + 15 * 60_000
    assert rt._llm_paused_until_ms == pytest.approx(expected, abs=2_000)
    assert rt._next_check_ms == rt._llm_paused_until_ms
    error = journal.recent("error", 1)[0]
    assert error["what"] == "llm usage limit" and error["resume"].endswith("UTC")


async def test_notes_from_an_earlier_session_are_not_shown(tmp_path):
    llm = ScriptedLLM([[turn(call("finish_tick", next_check_seconds=120, note="fresh"))]])
    rt, *_, journal = runtime(tmp_path, llm)
    journal.write("note", n=1, focus=None, text="old rule from yesterday")
    rt.started_ms = now_ms() + 1  # the earlier note was written before this session
    await rt.tick("start")
    first_user = llm.log[0][1]
    assert "old rule from yesterday" not in first_user and "(none yet)" in first_user


def test_prompt_includes_positioning_only_when_enabled():
    with_data = render_prompt(Path("prompts/trader.md"), RiskLimits(), 5.5, "demo")
    without = render_prompt(Path("prompts/trader.md"), RiskLimits(), 5.5, "demo", positioning=False)
    assert "`open_interest`" in with_data and "`funding`" in with_data
    assert "open_interest" not in without and "funding" not in without
    assert "$" not in without and "стены на пути к цели.\n\n" in without


async def test_focus_view_hides_positioning_when_disabled(tmp_path):
    rt, *_ = runtime(tmp_path, ScriptedLLM([]), AgentConfig(show_positioning=False))
    calls = []

    async def spy(symbol):
        calls.append(symbol)
        return {"x": 1}

    rt._funding = rt._open_interest = spy
    rt.focus_symbol = "SOL/USDT:USDT"
    view = await rt.focus_view()
    assert "funding" not in view and "open_interest" not in view and calls == []
