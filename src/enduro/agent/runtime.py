"""The agent loop: two modes, event-driven wake-ups, one short LLM episode per tick.

search mode — look at the radar, decide whether a coin deserves focus.
focus mode  — live view of one coin; enter, manage, exit, possibly many times in both
              directions, until the coin stops being readable; then release focus.

Each tick is a fresh episode: the stable system prompt and tools (cached) plus a user
message with the current state and the agent's recent notes. The agent schedules its
next check itself; a watchdog wakes it earlier if the price jumps or the position is
closed by the exchange (stop / take profit hit).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from enduro.agent.llm import (
    ApiLoopBackend,
    EpisodeBackend,
    LLMClient,
    LLMError,
    LLMTurn,
    ToolResult,
)
from enduro.agent.tools import TOOLS, TOOLS_BY_NAME, ToolInputError, to_json
from enduro.analytics.focus import FocusTracker
from enduro.analytics.radar_runner import RadarRunner
from enduro.core.models import MINUTE_MS, Candle, now_ms
from enduro.data.base import MarketDataSource
from enduro.data.collector import Collector
from enduro.journal.journal import Journal
from enduro.trading.service import TradingService

log = logging.getLogger(__name__)


class _Done(Exception):
    """Raised by the tick loop to stop all agent tasks after `max_ticks`."""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    search_interval_s: int = 180  # default next check when the agent does not schedule one
    focus_interval_s: int = 60
    max_llm_calls_per_tick: int = 8
    max_tool_calls_per_tick: int = 16
    wake_move_bps: float = 30.0  # wake early when the focus price moves this much
    notes_in_context: int = 10
    feedback_in_context: int = 15
    radar_rows_in_context: int = 10


class AgentRuntime:
    def __init__(
        self,
        llm: LLMClient | EpisodeBackend,
        system_prompt: str,
        trading: TradingService,
        radar: RadarRunner,
        focus_collector: Collector,
        focus_tracker: FocusTracker,
        reference_source: MarketDataSource,
        journal: Journal,
        config: AgentConfig,
        universe: Sequence[str],
        reference: str,
        execution: str,
        focus_events: asyncio.Queue | None = None,
    ) -> None:
        # A plain LLM client gets our own tool loop; a backend (e.g. Claude Code) runs its own.
        self.backend: EpisodeBackend = (
            llm
            if hasattr(llm, "run_episode")
            else ApiLoopBackend(llm, config.max_llm_calls_per_tick)
        )
        self._tool_calls_this_tick = 0
        self.system_prompt = system_prompt
        self.trading = trading
        self.radar = radar
        self.collector = focus_collector
        self.tracker = focus_tracker
        self.reference_source = reference_source
        self.journal = journal
        self.config = config
        self.universe = set(universe)
        self.reference = reference
        self.execution = execution
        self._focus_events = focus_events

        self.focus_symbol: str | None = None
        self.tick_no = 0
        self.session_cost_usd = 0.0
        self._next_check_ms = 0
        self._tick_note: str | None = None
        self._wake = asyncio.Event()
        self._wake_reason = "start"
        self._last_tick_mid: float | None = None
        self._had_position = False

    # ------------------------------------------------------------ views for tools

    def require_known_symbol(self, symbol: str) -> None:
        if symbol not in self.universe:
            raise ToolInputError(f"{symbol!r} is not in the scanned universe (use radar symbols)")

    def radar_view(self, top: int) -> dict[str, Any]:
        age_s = (now_ms() - self.radar.updated_ms) / 1000 if self.radar.updated_ms else None
        return {
            "universe_size": len(self.universe),
            "updated_s_ago": None if age_s is None else round(age_s),
            "rows": [r.to_summary() for r in self.radar.rows[:top]],
        }

    def focus_view(self) -> dict[str, Any]:
        assert self.focus_symbol is not None
        snap = self.tracker.snapshot(self.focus_symbol, now_ms())
        if snap is None:
            return {"symbol": self.focus_symbol, "status": "waiting for the first live data"}
        return snap.to_summary()

    async def price_history(self, symbol: str, interval: str, bars: int) -> list[Candle]:
        minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}[interval]
        since = now_ms() - (bars + 1) * minutes * MINUTE_MS
        candles = await self.reference_source.fetch_candles(symbol, since, bars + 1, interval)
        closed = [c for c in candles if c.ts + minutes * MINUTE_MS <= now_ms()]
        return closed[-bars:]

    async def has_open_position(self, symbol: str) -> bool:
        account = await self.trading.account()
        return any(p["symbol"] == symbol for p in account["positions"])

    async def set_focus(self, symbol: str, reason: str) -> None:
        if self.focus_symbol and self.focus_symbol != symbol:
            if await self.has_open_position(self.focus_symbol):
                raise ToolInputError(
                    f"close the position on {self.focus_symbol} before switching focus"
                )
            self.tracker.drop(self.focus_symbol)
        self.focus_symbol = symbol
        await self.collector.set_symbols([symbol])
        self._last_tick_mid = None
        self.journal.write("focus", symbol=symbol, reason=reason)

    async def release_focus(self, reason: str) -> None:
        if self.focus_symbol:
            self.tracker.drop(self.focus_symbol)
        self.journal.write("focus", symbol=None, released=self.focus_symbol, reason=reason)
        self.focus_symbol = None
        await self.collector.set_symbols([])

    def finish_tick(self, seconds: int, note: str) -> None:
        self._next_check_ms = now_ms() + seconds * 1000
        self._tick_note = note

    # ------------------------------------------------------------ main loop

    async def run(self, max_ticks: int | None = None) -> None:
        await self.radar.refresh()  # the first tick needs a fresh radar
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.radar.run_forever())
                tg.create_task(self.collector.run())
                tg.create_task(self._watchdog())
                tg.create_task(self._ticks(max_ticks))
                if self._focus_events is not None:
                    tg.create_task(self._consume_focus_events(self._focus_events))
        except* _Done:
            pass

    async def _ticks(self, max_ticks: int | None) -> None:
        while max_ticks is None or self.tick_no < max_ticks:
            await self.tick(self._wake_reason)
            self._wake.clear()
            self._wake_reason = "scheduled"
            timeout = max(0.0, (self._next_check_ms - now_ms()) / 1000)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout)
            except TimeoutError:
                pass
        raise _Done  # stops the background tasks too

    async def _consume_focus_events(self, queue: asyncio.Queue) -> None:
        while True:
            self.tracker.on_event(await queue.get())

    async def _watchdog(self) -> None:
        """Wake the agent early on a sharp move or when the exchange closed the position."""
        polls = 0
        while True:
            await asyncio.sleep(2)
            polls += 1
            if self.focus_symbol is None or self._wake.is_set():
                continue
            book = self.tracker.snapshot(self.focus_symbol, now_ms())
            mid = book.book[self.execution].mid if book and self.execution in book.book else None
            if mid and self._last_tick_mid:
                move = abs(mid / self._last_tick_mid - 1) * 1e4
                if move >= self.config.wake_move_bps:
                    self._wake_reason = f"price moved {move:.0f} bps since the last check"
                    self._wake.set()
                    continue
            if self._had_position and polls % 5 == 0:  # positions: every ~10 s
                if not await self.has_open_position(self.focus_symbol):
                    self._wake_reason = "position closed on the exchange (stop or take profit)"
                    self._wake.set()

    async def tick(self, trigger: str) -> None:
        self.tick_no += 1
        mode = "focus" if self.focus_symbol else "search"
        self._tick_note = None
        self._next_check_ms = 0
        self.journal.write(
            "tick", n=self.tick_no, mode=mode, focus=self.focus_symbol, trigger=trigger
        )
        try:
            await self._episode(await self._situation(mode, trigger))
        except LLMError as e:
            log.error("tick %d: %s", self.tick_no, e)
            self.journal.write("error", what="llm", error=str(e))
        default_s = (
            self.config.focus_interval_s if self.focus_symbol else self.config.search_interval_s
        )
        if not self._next_check_ms:
            self._next_check_ms = now_ms() + default_s * 1000
        if self._tick_note:
            self.journal.write(
                "note", n=self.tick_no, focus=self.focus_symbol, text=self._tick_note
            )
        log.info(
            "tick %d [%s] next in %ds, session cost $%.3f | %s",
            self.tick_no,
            self.focus_symbol or "search",
            (self._next_check_ms - now_ms()) // 1000,
            self.session_cost_usd,
            (self._tick_note or "(no note)").replace("\n", " ")[:300],
        )
        await self._remember_market_state()

    async def _remember_market_state(self) -> None:
        if self.focus_symbol:
            snap = self.tracker.snapshot(self.focus_symbol, now_ms())
            if snap and self.execution in snap.book:
                self._last_tick_mid = snap.book[self.execution].mid
            self._had_position = await self.has_open_position(self.focus_symbol)
        else:
            self._had_position = False

    async def _situation(self, mode: str, trigger: str) -> str:
        account = await self.trading.account()
        notes = self.journal.recent("note", self.config.notes_in_context)
        parts = [
            f"Tick {self.tick_no} · {datetime.now(UTC):%Y-%m-%d %H:%M:%S} UTC · mode: {mode}"
            f" · trigger: {trigger}",
            "",
            "## Account and risk",
            json.dumps(account, ensure_ascii=False),
        ]
        if mode == "focus":
            parts += [
                "",
                f"## Focus: {self.focus_symbol}",
                json.dumps(self.focus_view(), ensure_ascii=False),
            ]
        else:
            parts += [
                "",
                "## Radar",
                json.dumps(self.radar_view(self.config.radar_rows_in_context), ensure_ascii=False),
            ]
        parts += ["", "## Your recent notes (oldest first)"]
        parts += [
            f"- {datetime.fromtimestamp(n['ts'] / 1000, UTC):%H:%M:%S} "
            f"[{n.get('focus') or 'search'}] {n['text']}"
            for n in notes
        ] or ["- (none yet)"]
        reported = self.journal.recent("feedback", self.config.feedback_in_context)
        if reported:
            parts += ["", "## Tooling gaps you already reported (don't repeat them)"]
            parts += [f"- [{r['category']}] {r['title']}" for r in reported]
        parts += ["", "Decide what to do now. End the check with finish_tick."]
        return "\n".join(parts)

    async def _episode(self, situation: str) -> None:
        self._tool_calls_this_tick = 0
        text = await self.backend.run_episode(
            self.system_prompt,
            situation,
            [t.spec for t in TOOLS],
            self.call_tool,
            self._account_llm,
            lambda: self._tick_note is not None,
        )
        if self._tick_note is None and text:  # ended without finish_tick: keep its words
            self._tick_note = text.strip()[:2000]

    async def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute one tool call for whichever backend runs the episode."""
        return await self._run_tool(name, "", args)

    async def _run_tool(self, name: str, call_id: str, args: dict[str, Any]) -> ToolResult:
        tool = TOOLS_BY_NAME.get(name)
        self._tool_calls_this_tick += 1
        try:
            if tool is None:
                raise ToolInputError(f"unknown tool {name!r}")
            if (
                self._tool_calls_this_tick > self.config.max_tool_calls_per_tick
                and name != "finish_tick"
            ):
                raise ToolInputError("tool budget for this check is used up: call finish_tick")
            if self._tick_note is not None and name != "finish_tick":
                raise ToolInputError("this check is finished (finish_tick was called); stop here")
            result = await tool.handler(self, args)
            self.journal.write("tool", n=self.tick_no, name=name, input=args, result=result)
            return ToolResult(call_id, to_json(result))
        except ToolInputError as e:
            self.journal.write("tool", n=self.tick_no, name=name, input=args, error=str(e))
            return ToolResult(call_id, f"error: {e}", is_error=True)
        except Exception as e:  # exchange / network failures: tell the model, keep going
            log.exception("tool %s failed", name)
            self.journal.write("tool", n=self.tick_no, name=name, input=args, error=repr(e))
            return ToolResult(call_id, f"error: {type(e).__name__}: {e}", is_error=True)

    def _account_llm(self, turn: LLMTurn) -> None:
        self.session_cost_usd += turn.usage.cost_usd
        self.journal.write(
            "llm",
            n=self.tick_no,
            model=turn.model,
            stop_reason=turn.stop_reason,
            usage=turn.usage,
            text=turn.text,
            tool_calls=[{"name": c.name, "input": c.input} for c in turn.tool_calls],
        )
