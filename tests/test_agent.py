import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from enduro.agent.llm import LLMTurn, ToolCall, ToolResult, Usage
from enduro.agent.prompt import render_prompt
from enduro.agent.runtime import AgentConfig, AgentRuntime, wake_threshold_bps
from enduro.agent.tools import TOOLS
from enduro.analytics.focus import FocusTracker
from enduro.analytics.liquidity import Liquidity, LiquidityBook, liquidity_from_book
from enduro.core.models import OrderBook
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


class FakeRadar:
    updated_ms = 1

    def __init__(self) -> None:
        self.rows: list = []
        self.liquidity = LiquidityBook(min_depth_usd=2_000, max_spread_bps=10)

    async def refresh(self):
        return []


class FakeCollector:
    def __init__(self) -> None:
        self.symbols: list[str] = []

    async def set_symbols(self, symbols):
        self.symbols = list(symbols)


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
    focus = rt.focus_view()
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
