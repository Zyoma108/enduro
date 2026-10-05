"""The agent loop: two modes, event-driven wake-ups, one short LLM episode per tick.

search mode — look at the radar, decide whether a coin deserves focus.
focus mode  — live view of one coin; enter, manage, exit, possibly many times in both
              directions, until the coin stops being readable; then release focus.

Each tick is a fresh episode: the stable system prompt and tools (cached) plus a user
message with the current state and the agent's recent notes. The agent schedules its
next check itself; a watchdog wakes it earlier if the price jumps or the position is
closed by the exchange (stop / take profit hit), and a price alert the agent left on any
coin wakes it when a 1m candle closes beyond the level.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from enduro.agent.alerts import AlertBook, FiredAlert
from enduro.agent.llm import (
    ApiLoopBackend,
    EpisodeBackend,
    LLMClient,
    LLMError,
    LLMTurn,
    ToolResult,
)
from enduro.agent.review import HOLD_CHECK_MIN, TradeContext, review_trade, trade_context
from enduro.agent.tools import TOOLS, TOOLS_BY_NAME, ToolInputError, to_json
from enduro.analytics.focus import HISTORY_MS, FocusTracker
from enduro.analytics.metrics import Bars
from enduro.analytics.radar import RadarRow
from enduro.analytics.radar_runner import RadarRunner
from enduro.analytics.stops import stop_context
from enduro.core.models import MINUTE_MS, Candle, Funding, now_ms
from enduro.data.base import MarketDataSource
from enduro.data.collector import Collector
from enduro.execution.models import ClosedTrade
from enduro.journal.journal import Journal
from enduro.trading.service import TradingService

log = logging.getLogger(__name__)

RETRY_AFTER_ERROR_S = 30  # a tick that failed (model or exchange down) is retried this soon
BACKFILL_TIMEOUT_S = 10  # per exchange; paging itself stops after ~5 s
CLOSED_TRADES_FETCHED = 10
CLOSED_TRADES_IN_CONTEXT = 5
CLOSED_TRADES_HORIZON_MS = 24 * 3_600_000  # older closes are not journaled late
CONTEXT_TEXT_CHARS = 300  # thesis / exit reason quoted back with a closed trade
HINDSIGHT_REFRESH_MS = 60_000
FUNDING_REFRESH_MS = 60_000
FUNDING_TIMEOUT_S = 5
FUNDING_NOTE = (
    "rate_pct — ставка ближайшего расчёта (оценка биржи сейчас), + значит лонги платят "
    "шортам; платит или получает позиция, открытая на Bybit в момент расчёта (next_in_min)"
)


def funding_summary(f: Funding, now: int) -> dict[str, Any]:
    out: dict[str, Any] = {"rate_pct": round(f.rate * 100, 4), "interval_h": f.interval_h}
    if f.interval_h:
        out["per_day_pct"] = round(f.rate * 100 * 24 / f.interval_h, 4)
    if f.next_ts:
        out["next_in_min"] = max(0, round((f.next_ts - now) / MINUTE_MS))
    return out


def wake_threshold_bps(atr_5m: float | None, in_position: bool, config: AgentConfig) -> float:
    """How far the price must move since the last check to wake the agent early."""
    threshold = config.wake_move_bps
    if atr_5m and atr_5m > 0:
        threshold = max(threshold, config.wake_move_atr * atr_5m * 1e4)
    return threshold if in_position else threshold * config.wake_flat_multiplier


class _Done(Exception):
    """Raised by the tick loop to stop all agent tasks when the session is over."""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    search_interval_s: int = 180  # default next check when the agent does not schedule one
    focus_interval_s: int = 60
    max_llm_calls_per_tick: int = 8
    max_tool_calls_per_tick: int = 16
    # Early wake-up on a price move: half a 5m ATR of the coin (never below the floor);
    # without a position the bar is doubled and wake-ups are spaced further apart.
    wake_move_bps: float = 30.0  # floor
    wake_move_atr: float = 0.5
    wake_flat_multiplier: float = 2.0
    min_wake_gap_flat_s: int = 60
    min_wake_gap_position_s: int = 15
    min_check_flat_s: int = 120  # no position: scheduled checks no more often than this
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
        alerts: AlertBook | None = None,
        execution_source: MarketDataSource | None = None,
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
        # Prices of the exchange we trade on, for hindsight on closed trades.
        self.execution_source = execution_source
        self._funding_cache: dict[str, tuple[int, dict[str, Any]]] = {}
        # order id -> (hindsight, final, fetched_ms)
        self._hindsight: dict[str, tuple[dict[str, Any], bool, int]] = {}
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
        self._last_tick_end_ms = 0
        self._had_position = False
        self.alerts = alerts or AlertBook()
        self._fired_alerts: list[FiredAlert] = []
        self._in_tick = False
        # Tick limit reached with a position open: keep managing it, no new entries.
        self.wind_down = False
        self._known_closed: set[str] | None = None  # closing order ids already journaled

    # ------------------------------------------------------------ views for tools

    def require_known_symbol(self, symbol: str) -> None:
        if symbol not in self.universe:
            raise ToolInputError(f"{symbol!r} is not in the scanned universe (use radar symbols)")

    def radar_view(self, top: int) -> dict[str, Any]:
        """Radar rows the agent can act on: coins too illiquid on the execution exchange
        are left out (only counted), the rest carry their spread and depth there."""
        age_s = (now_ms() - self.radar.updated_ms) / 1000 if self.radar.updated_ms else None
        liquidity = self.radar.liquidity
        tradable = [r for r in self.radar.rows if liquidity.tradable(r.symbol) is not False]
        return {
            "universe_size": len(self.universe),
            "updated_s_ago": None if age_s is None else round(age_s),
            "hidden_illiquid_on_execution_exchange": len(self.radar.rows) - len(tradable),
            "liquidity_rule": (
                f"{self.execution} spread <= {liquidity.max_spread_bps:g} bps and depth within "
                f"10 bps >= {liquidity.min_depth_usd:,.0f} USDT on the thinner side"
            ),
            "rows": [{**r.to_summary(), **liquidity.summary(r.symbol)} for r in tradable[:top]],
        }

    def radar_row(self, symbol: str) -> RadarRow | None:
        return next((r for r in self.radar.rows if r.symbol == symbol), None)

    def atr_5m(self, symbol: str) -> float | None:
        row = self.radar_row(symbol)
        atr = row.atr_5m if row else None
        return atr if atr is not None and atr == atr else None  # NaN -> None

    def stop_context(self, side: str, minutes: int = 120) -> dict[str, Any]:
        assert self.focus_symbol is not None
        symbol = self.focus_symbol
        candles = self.radar.radar.candles(symbol, minutes) if self.radar.radar else []
        if len(candles) < 30:
            raise ToolInputError(f"not enough 1m history for {symbol} yet ({len(candles)} bars)")
        # Levels come from reference candles; stops trigger on execution prices, which sit
        # a basis away. Measure on the reference and convert prices by the live basis.
        ref = self.tracker.latest_book(self.reference, symbol)
        exe = self.tracker.latest_book(self.execution, symbol)
        price = ref.mid if ref and ref.mid else candles[-1].close
        scale = exe.mid / ref.mid if ref and ref.mid and exe and exe.mid else 1.0
        context = stop_context(
            Bars.from_candles(candles), price, side, self.atr_5m(symbol), price_scale=scale
        )
        if scale != 1.0:
            context["prices_on"] = self.execution
            context["basis_bps"] = round((scale - 1) * 1e4, 1)
            context["source"] = (
                f"levels from {self.reference} 1m candles, shifted to {self.execution} prices "
                "by the live basis: set stops from these prices directly"
            )
        else:
            context["prices_on"] = self.reference
            context["source"] = (
                f"{self.reference} 1m candles; no live books for the basis yet, so prices are "
                f"{self.reference} prices — {self.execution} may differ by a few bps"
            )
        return context

    async def focus_view(self) -> dict[str, Any]:
        assert self.focus_symbol is not None
        symbol = self.focus_symbol
        atr = self.atr_5m(symbol)
        context: dict[str, Any] = {
            # Typical 5-minute range: the noise a stop has to survive.
            "atr_5m_pct": None if atr is None else round(atr * 100, 3),
            **self.radar.liquidity.summary(symbol),
        }
        if funding := await self._funding(symbol):
            context["funding"] = funding
        snap = self.tracker.snapshot(symbol, now_ms())
        if snap is None:
            return {"symbol": symbol, "status": "waiting for the first live data", **context}
        return {**snap.to_summary(), **context}

    async def _funding(self, symbol: str) -> dict[str, Any] | None:
        """Current funding on both exchanges, refreshed at most once a minute. An exchange
        that does not answer in time is left out."""
        now = now_ms()
        cached = self._funding_cache.get(symbol)
        if cached and now - cached[0] < FUNDING_REFRESH_MS:
            return cached[1]
        sources = {
            ex: src
            for ex, src in (
                (self.execution, self.execution_source),
                (self.reference, self.reference_source),
            )
            if src is not None
        }
        if not sources:
            return None

        async def fetch(src: MarketDataSource) -> Funding:
            async with asyncio.timeout(FUNDING_TIMEOUT_S):
                return await src.fetch_funding(symbol)

        results = await asyncio.gather(
            *(fetch(s) for s in sources.values()), return_exceptions=True
        )
        view: dict[str, Any] = {}
        for ex, result in zip(sources, results, strict=True):
            if isinstance(result, BaseException):
                log.warning("%s funding of %s unavailable: %r", ex, symbol, result)
                continue
            view[ex] = funding_summary(result, now)
        if not view:
            return cached[1] if cached else None
        view["note"] = FUNDING_NOTE
        self._funding_cache[symbol] = (now, view)
        return view

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
        refocus = self.focus_symbol == symbol
        self.focus_symbol = symbol
        await self.collector.set_symbols([symbol])
        self._last_tick_mid = None
        self.journal.write("focus", symbol=symbol, reason=reason)
        if not refocus:
            await self._backfill_focus(symbol)

    async def _backfill_focus(self, symbol: str) -> None:
        """Load the last minutes of trades so the agent sees the flow at once instead of
        a few seconds of it. Best effort: without it the windows just fill up live."""
        by_exchange = await self.collector.recent_trades(
            symbol, now_ms() - HISTORY_MS, timeout_s=BACKFILL_TIMEOUT_S
        )
        if self.focus_symbol != symbol:
            return  # focus moved on meanwhile
        for exchange, trades in by_exchange.items():
            added = self.tracker.backfill(exchange, symbol, trades)
            span_s = (trades[-1].ts - trades[0].ts) / 1000 if trades else 0
            log.info("backfilled %s %s: %d trades over %.0fs", exchange, symbol, added, span_s)

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
                tg.create_task(self.radar.run_forever(lambda _rows: self._check_alerts()))
                tg.create_task(self.collector.run())
                tg.create_task(self._watchdog())
                tg.create_task(self._ticks(max_ticks))
                if self._focus_events is not None:
                    tg.create_task(self._consume_focus_events(self._focus_events))
        except* _Done:
            pass

    async def _ticks(self, max_ticks: int | None) -> None:
        while True:
            self._in_tick = True
            try:
                await self.tick(self._wake_reason)
            finally:
                self._in_tick = False
            self._wake.clear()
            self._wake_reason = "scheduled"
            self._wake_on_fired_alerts()  # alerts that fired while the tick was running
            if await self._session_over(max_ticks):
                break
            timeout = max(0.0, (self._next_check_ms - now_ms()) / 1000)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout)
            except TimeoutError:
                pass
            if self.wind_down and await self._session_over(max_ticks):
                break  # the exchange closed the position (stop / take profit) meanwhile
        raise _Done  # stops the background tasks too

    async def _session_over(self, max_ticks: int | None) -> bool:
        """After `max_ticks` the session ends — but never with a position left unmanaged:
        until it is closed the agent keeps ticking, only to manage it."""
        if max_ticks is None or self.tick_no < max_ticks:
            return False
        try:
            positions = (await self.trading.account())["positions"]
        except Exception:
            log.exception("cannot check positions at the tick limit; continuing")
            return False
        if not positions:
            return True
        if not self.wind_down:
            self.wind_down = True
            held = ", ".join(f"{p['symbol']} {p['side']}" for p in positions)
            self.journal.write(
                "session",
                text=f"tick limit {max_ticks} reached with an open position ({held}): "
                "managing it until it is closed, no new entries",
            )
        return False

    def _check_alerts(self) -> None:
        """After each radar refresh (a new closed 1m candle): fire crossed alerts."""
        radar = self.radar.radar
        if radar is None:
            return
        now = now_ms()
        for alert in self.alerts.prune(now):
            self.journal.write("alert", action="expired", alert=alert)
        fired = self.alerts.check(lambda symbol: radar.candles(symbol, 5), now)
        for f in fired:
            self.journal.write("alert", action="fired", alert=f.alert, close=f.candle.close)
        self._fired_alerts += fired
        self._wake_on_fired_alerts()

    def _wake_on_fired_alerts(self) -> None:
        # A wake-up set during a tick would be cleared when it ends: keep them until then.
        if not self._fired_alerts or self._in_tick:
            return
        self._wake_reason = "; ".join(f.describe() for f in self._fired_alerts)
        self._fired_alerts = []
        self._wake.set()

    async def _consume_focus_events(self, queue: asyncio.Queue) -> None:
        while True:
            self.tracker.on_event(await queue.get())

    async def _watchdog(self) -> None:
        """Wake the agent early on a sharp move or when the exchange closed the position.
        Survives exchange/network errors: a failed poll is retried on the next one."""
        polls = 0
        while True:
            await asyncio.sleep(2)
            polls += 1
            if self.focus_symbol is None or self._wake.is_set() or self._in_tick:
                continue
            try:
                await self._watch_once(polls)
            except Exception as e:
                if polls % 15 == 0:  # don't flood the log during an outage
                    log.warning("watchdog poll failed: %s: %s", type(e).__name__, e)

    async def _watch_once(self, polls: int) -> None:
        assert self.focus_symbol is not None
        gap_s = (
            self.config.min_wake_gap_position_s
            if self._had_position
            else self.config.min_wake_gap_flat_s
        )
        book = self.tracker.latest_book(self.execution, self.focus_symbol)
        mid = book.mid if book else None
        if mid and self._last_tick_mid and now_ms() - self._last_tick_end_ms >= gap_s * 1000:
            move = abs(mid / self._last_tick_mid - 1) * 1e4
            threshold = wake_threshold_bps(
                self.atr_5m(self.focus_symbol), self._had_position, self.config
            )
            if move >= threshold:
                self._wake_reason = (
                    f"price moved {move:.0f} bps since the last check "
                    f"(wake threshold {threshold:.0f} bps)"
                )
                self._wake.set()
                return
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
            self._next_check_ms = self._next_check_ms or now_ms() + RETRY_AFTER_ERROR_S * 1000
        except Exception as e:  # exchange / network down: try again soon, don't die
            log.error("tick %d failed: %s: %s", self.tick_no, type(e).__name__, e)
            self.journal.write("error", what="tick", error=f"{type(e).__name__}: {e}")
            self._next_check_ms = self._next_check_ms or now_ms() + RETRY_AFTER_ERROR_S * 1000
        default_s = (
            self.config.focus_interval_s if self.focus_symbol else self.config.search_interval_s
        )
        if not self._next_check_ms:
            self._next_check_ms = now_ms() + default_s * 1000
        try:
            await self._remember_market_state()
        except Exception as e:  # never let bookkeeping after the episode kill the agent
            log.error("tick %d: market state: %s: %s", self.tick_no, type(e).__name__, e)
            self.journal.write("error", what="market state", error=f"{type(e).__name__}: {e}")
        if not self._had_position:
            # Flat: scheduled checks are spaced out to save model usage; a sharp move
            # still wakes the agent through the watchdog.
            self._next_check_ms = max(
                self._next_check_ms, now_ms() + self.config.min_check_flat_s * 1000
            )
        next_check_s = (self._next_check_ms - now_ms()) // 1000
        if self._tick_note:
            self.journal.write(
                "note",
                n=self.tick_no,
                focus=self.focus_symbol,
                text=self._tick_note,
                next_check_s=next_check_s,
                session_cost_usd=round(self.session_cost_usd, 4),
            )
        else:
            log.info(
                "tick %d [%s] ended without a note; next in %ds, session cost $%.3f",
                self.tick_no,
                self.focus_symbol or "search",
                next_check_s,
                self.session_cost_usd,
            )

    async def _remember_market_state(self) -> None:
        self._last_tick_end_ms = now_ms()
        if self.focus_symbol:
            snap = self.tracker.snapshot(self.focus_symbol, now_ms())
            if snap and self.execution in snap.book:
                self._last_tick_mid = snap.book[self.execution].mid
            try:
                self._had_position = await self.has_open_position(self.focus_symbol)
            except Exception as e:  # keep the last known state; the watchdog retries
                log.warning("cannot check the position: %s: %s", type(e).__name__, e)
        else:
            self._had_position = False

    async def _sync_closed_trades(self) -> list[dict[str, Any]]:
        """Journal positions the exchange closed since we last looked — stops, take
        profits, and anything closed while the agent was not running — and return the
        latest few for the agent to see."""
        trades = await self.trading.gateway.closed_trades(CLOSED_TRADES_FETCHED)
        if self._known_closed is None:
            self._known_closed = {
                (r.get("trade") or {}).get("order_id") for r in self.journal.recent("closed", 1000)
            }
        orders = self.journal.recent("order", 1000)
        closes = {
            (r.get("result") or {}).get("id"): r for r in orders if r.get("action") == "close"
        }
        horizon = now_ms() - CLOSED_TRADES_HORIZON_MS
        for t in reversed(trades):  # oldest first, so the journal reads in order
            by = "agent" if t.order_id in closes else "exchange: stop loss / take profit"
            if t.order_id not in self._known_closed and t.closed_ms >= horizon:
                self.journal.write("closed", trade=t, closed_by=by)
            self._known_closed.add(t.order_id)

        risks = self.journal.recent("risk", 1000)
        summaries = []
        for t in trades[:CLOSED_TRADES_IN_CONTEXT]:  # newest first
            close = closes.get(t.order_id)
            summary: dict[str, Any] = {
                "closed_utc": f"{datetime.fromtimestamp(t.closed_ms / 1000, UTC):%H:%M:%S}",
                "symbol": t.symbol,
                "side": t.side,
                "qty": t.qty,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "pnl_usdt_net_of_fees": round(t.pnl, 2),
                "closed_by": "agent" if close else "exchange: stop loss / take profit",
            }
            context = trade_context(t, orders, risks)
            if context.thesis:
                summary["your_thesis"] = context.thesis[:CONTEXT_TEXT_CHARS]
            if close and close.get("reason"):
                summary["your_exit_reason"] = close["reason"][:CONTEXT_TEXT_CHARS]
            if context.stop is not None:
                summary["stop_at_exit"] = context.stop
            if context.take is not None:
                summary["take_profit_at_exit"] = context.take
            hindsight = await self._trade_hindsight(t, context, by_agent=close is not None)
            if hindsight is not None:
                summary["hindsight"] = hindsight
            summaries.append(summary)
        return summaries

    async def _trade_hindsight(
        self, trade: ClosedTrade, context: TradeContext, by_agent: bool
    ) -> dict[str, Any] | None:
        """Hindsight from the execution exchange's candles, cached; refreshed at most once
        a minute until the hour after the exit is complete."""
        source = self.execution_source or self.reference_source
        if source is None:
            return None
        cached = self._hindsight.get(trade.order_id)
        now = now_ms()
        if cached and (cached[1] or now - cached[2] < HINDSIGHT_REFRESH_MS):
            return cached[0]
        start = context.opened_ms if context.opened_ms is not None else trade.closed_ms
        since = start // MINUTE_MS * MINUTE_MS
        until = min(now, trade.closed_ms + (HOLD_CHECK_MIN + 1) * MINUTE_MS)
        bars = min(1000, (until - since) // MINUTE_MS + 2)
        try:
            candles = await source.fetch_candles(trade.symbol, since, bars)
        except Exception:
            log.warning("hindsight for %s unavailable", trade.symbol, exc_info=True)
            return cached[0] if cached else None
        closed_bars = [c for c in candles if c.ts + MINUTE_MS <= now]
        hindsight, final = review_trade(
            side=trade.side,
            entry=trade.entry_price,
            exit_price=trade.exit_price,
            opened_ms=context.opened_ms,
            closed_ms=trade.closed_ms,
            stop=context.stop,
            take=context.take,
            by_agent=by_agent,
            candles=closed_bars,
            now_ms=now,
        )
        self._hindsight[trade.order_id] = (hindsight, final, now)
        return hindsight

    async def _situation(self, mode: str, trigger: str) -> str:
        account = await self.trading.account()
        try:
            closed = await self._sync_closed_trades()
        except Exception:
            log.warning("cannot read closed trades from the exchange", exc_info=True)
            closed = None
        notes = self.journal.recent("note", self.config.notes_in_context)
        parts = [
            f"Tick {self.tick_no} · {datetime.now(UTC):%Y-%m-%d %H:%M:%S} UTC · mode: {mode}"
            f" · trigger: {trigger}",
            "",
        ]
        if self.wind_down:
            parts += [
                "## Session ending",
                "The session's tick limit is reached. Manage the open position to its end "
                "(stop, take profit or your exit) as usual; no new entries. The session "
                "stops once you are flat.",
                "",
            ]
        parts += [
            "## Account and risk",
            json.dumps(account, ensure_ascii=False),
        ]
        if closed:
            parts += [
                "",
                "## Recently closed positions (exchange records, newest first)",
                "hindsight (минутные свечи Bybit): best/worst_while_open_pct — насколько цена "
                "уходила в твою пользу и против тебя, пока позиция была открыта; "
                "after_exit_pct — где цена была потом, + значит она продолжила идти в сторону "
                "сделки после твоего выхода; if_held — для твоих ручных выходов: что цена "
                "задела бы раньше, твой стоп или тейк.",
            ]
            parts += [json.dumps(c, ensure_ascii=False) for c in closed]
        if mode == "focus":
            parts += [
                "",
                f"## Focus: {self.focus_symbol}",
                json.dumps(await self.focus_view(), ensure_ascii=False),
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
        alerts = self.alerts.active(now_ms())
        parts += ["", "## Your price alerts"]
        parts += [json.dumps(a.to_summary(now_ms()), ensure_ascii=False) for a in alerts] or [
            "- (none)"
        ]
        reported = self.journal.recent("feedback", self.config.feedback_in_context)
        if reported:
            parts += ["", "## Tooling gaps you already reported (don't repeat them)"]
            parts += [f"- [{r['category']}] {r['title']}" for r in reported]
        # The journal is read by a Russian-speaking human: without yesterday's notes in
        # context (after a day off) the model drifted into English (2026-10-04).
        parts += [
            "",
            "Реши, что делать сейчас, и закончи проверку вызовом finish_tick. "
            "Заметки, тезисы и причины пиши по-русски.",
        ]
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
