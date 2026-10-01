"""Command-line entry point: `enduro <command>`."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import sys
from datetime import UTC, datetime

from enduro.analytics.focus import WINDOWS_S, FocusSnapshot, FocusTracker
from enduro.analytics.market_state import MarketState
from enduro.analytics.radar import RadarRow
from enduro.analytics.radar_runner import RadarRunner
from enduro.config import Settings
from enduro.core.models import OrderBook, now_ms
from enduro.data.bus import EventBus
from enduro.data.ccxt_source import CcxtSource
from enduro.data.collector import Collector
from enduro.data.history import backfill
from enduro.data.universe import UniverseEntry, build_universe
from enduro.execution.bybit import BybitGateway
from enduro.execution.models import OrderAction, OrderRequest, OrderResult, PositionSide
from enduro.risk.manager import RiskLimits, RiskManager, RiskStateStore
from enduro.storage import store
from enduro.storage.candles import last_candle_ts, write_candles
from enduro.storage.parquet_sink import ParquetSink

log = logging.getLogger(__name__)


def _cancel_on_shutdown_signals() -> None:
    """Turn SIGINT/SIGTERM into cancellation of the current task for a clean shutdown."""
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)


async def _collect(settings: Settings, interval_s: float) -> None:
    _cancel_on_shutdown_signals()
    market, storage = settings.market, settings.storage
    bus = EventBus()
    queue = bus.subscribe()
    state = MarketState()
    sources = [CcxtSource(ex, market_type=market.market_type) for ex in market.exchanges]
    collector = Collector(sources, market.symbols, bus, book_depth=market.book_depth)
    sink = None
    if storage.enabled:
        sink = ParquetSink(
            storage.root,
            flush_interval_s=storage.flush_interval_s,
            book_interval_ms=storage.book_snapshot_interval_ms,
        )
        log.info("recording market data to %s", storage.root.resolve())

    async def consume() -> None:
        while True:
            state.on_event(await queue.get())

    async def report() -> None:
        while True:
            await asyncio.sleep(interval_s)
            print(_render(state, settings, bus.dropped, sink), flush=True)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(collector.run())
        tg.create_task(consume())
        tg.create_task(report())
        if sink is not None:
            tg.create_task(sink.run(bus.subscribe(maxsize=100_000)))


async def _build_universe(
    settings: Settings, reference: CcxtSource, execution: CcxtSource
) -> list[UniverseEntry]:
    scanner = settings.scanner
    return await build_universe(
        reference, execution, scanner.min_quote_volume_usd, scanner.asset_classes
    )


async def _scan_with_signals(settings: Settings, top: int, as_json: bool, once: bool) -> None:
    _cancel_on_shutdown_signals()
    await _scan(settings, top, as_json, once)


def _gateway(settings: Settings) -> BybitGateway:
    api_key, api_secret = settings.bybit.require()
    return BybitGateway(
        api_key,
        api_secret,
        environment=settings.execution.environment,
        allow_live=settings.execution.allow_live,
    )


async def _account(settings: Settings, setup: bool) -> None:
    gateway = _gateway(settings)
    try:
        state = await (gateway.ensure_account_setup() if setup else gateway.account_state())
        balance = await gateway.balance()
        positions = await gateway.positions()
        orders = await gateway.open_orders()
    finally:
        await gateway.close()
    print(f"environment : {state.environment}")
    print(f"margin mode : {state.margin_mode}")
    print(f"hedge mode  : {state.hedge_mode}")
    print(f"equity      : {balance.equity:,.2f} USDT")
    print(f"available   : {balance.available:,.2f} USDT")
    print(f"positions   : {len(positions)}")
    for p in positions:
        print(
            f"  {p.symbol:<18} {p.side:<5} size {p.size:g} entry {p.entry_price} "
            f"mark {p.mark_price} uPnL {p.unrealized_pnl} lev {p.leverage}"
        )
    print(f"open orders : {len(orders)}")
    for o in orders:
        print(f"  {o.symbol:<18} {o.side:<4} {o.qty:g} @ {o.avg_price} [{o.status}] id={o.id}")
    if not setup and (state.margin_mode != "cross" or not state.hedge_mode):
        print("\naccount is not in cross margin + hedge mode; run `enduro account --setup`")


async def _test_trade(
    settings: Settings, symbol: str, side: PositionSide, stop_pct: float | None
) -> None:
    """Open the minimum size at market on demo, then close it; report fills vs the book."""
    if settings.execution.environment != "demo":
        sys.exit("test-trade only runs against the demo environment")
    gateway = _gateway(settings)
    market_data = CcxtSource(settings.market.execution_exchange)  # public mainnet book
    tag = f"enduro-test-{now_ms()}"
    started = finished = False
    try:
        state = await gateway.account_state()
        if state.margin_mode != "cross" or not state.hedge_mode:
            sys.exit("account is not in cross margin + hedge mode; run `enduro account --setup`")
        if any(p.side == side for p in await gateway.positions([symbol])):
            sys.exit(f"there is already a {side} position on {symbol}; not touching it")
        rules = await gateway.instrument_rules(symbol)
        book = await market_data.fetch_top_of_book(symbol)
        qty = rules.min_order_qty(book.best_ask or book.best_bid)
        print(
            f"{symbol}: qty step {rules.qty_step:g}, min qty {rules.min_qty:g}, "
            f"min notional {rules.min_notional:g} USDT -> order qty {qty:g}"
        )

        stop = None
        if stop_pct is not None:
            ref = book.best_ask if side == "long" else book.best_bid
            raw_stop = ref * (1 - stop_pct / 100 if side == "long" else 1 + stop_pct / 100)
            stop = round(round(raw_stop / rules.price_tick) * rules.price_tick, 10)

        async def execute(action: OrderAction, book_before: OrderBook) -> OrderResult:
            request = OrderRequest(
                symbol,
                side,
                action,
                qty,
                client_order_id=f"{tag}-{action[0]}",
                stop_loss=stop if action == "open" else None,
            )
            placed = await gateway.place_order(request)
            order = await gateway.wait_for_fill(placed.id, symbol)
            touch = book_before.best_ask if request.side == "buy" else book_before.best_bid
            slip = "—"
            if order.avg_price and touch:
                sign = 1 if request.side == "buy" else -1
                slip = f"{sign * (order.avg_price - touch) / touch * 1e4:+.2f} bps"
            print(
                f"{action:<5} {request.side:<4} {order.filled:g}/{order.qty:g} [{order.status}] "
                f"avg {order.avg_price} | bid {book_before.best_bid} ask {book_before.best_ask} "
                f"| vs touch {slip} | fee {order.fee}"
            )
            return order

        started = True  # from here on, a position on this side is ours to clean up
        opened = await execute("open", book)
        for p in await gateway.positions([symbol]):
            print(
                f"position: {p.side} {p.size:g} entry {p.entry_price} lev {p.leverage} "
                f"stop loss on exchange: {p.stop_loss} (requested {stop})"
            )
        closed = await execute("close", await market_data.fetch_top_of_book(symbol))
        left = [p for p in await gateway.positions([symbol]) if p.side == side]
        print(f"position after close: {left[0].size:g}" if left else "position after close: flat")
        finished = not left
        if opened.avg_price and closed.avg_price:
            sign = 1 if side == "long" else -1
            gross = sign * (closed.avg_price - opened.avg_price) * closed.filled
            fees = (opened.fee or 0.0) + (closed.fee or 0.0)
            print(f"PnL: gross {gross:+.4f} - fees {fees:.4f} = net {gross - fees:+.4f} USDT")
    finally:
        if started and not finished:
            # We checked the side was flat before starting, so whatever is there is ours.
            for p in await gateway.positions([symbol]):
                if p.side == side:
                    log.error(
                        "test trade did not finish cleanly, closing %s %s %g", symbol, side, p.size
                    )
                    await gateway.place_order(OrderRequest(symbol, side, "close", p.size))
        await asyncio.gather(gateway.close(), market_data.close())


FOCUS_BOOK_DEPTH = 1000
FOCUS_BOOK_LIMITS = {"bybit": 1000}  # Binance keeps a full local book by default


async def _focus(settings: Settings, symbols: list[str], interval_s: float, as_json: bool) -> None:
    _cancel_on_shutdown_signals()
    market = settings.market
    exchanges = [market.reference_exchange, market.execution_exchange]
    bus = EventBus()
    queue = bus.subscribe()
    tracker = FocusTracker(market.reference_exchange, market.execution_exchange)
    # Deep books: depth within ±25 bps and slippage for $50k need hundreds of levels.
    sources = [
        CcxtSource(ex, market_type=market.market_type, book_limit=FOCUS_BOOK_LIMITS.get(ex))
        for ex in exchanges
    ]
    collector = Collector(sources, symbols, bus, book_depth=FOCUS_BOOK_DEPTH)

    async def consume() -> None:
        while True:
            tracker.on_event(await queue.get())

    async def report() -> None:
        while True:
            await asyncio.sleep(interval_s)
            for symbol in symbols:
                snap = tracker.snapshot(symbol, now_ms())
                if snap is None:
                    continue
                if as_json:
                    print(json.dumps(snap.to_summary(), ensure_ascii=False), flush=True)
                else:
                    print(_render_focus(snap), flush=True)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(collector.run())
        tg.create_task(consume())
        tg.create_task(report())


def _render_focus(snap: FocusSnapshot) -> str:
    def f(x: float | None, fmt: str) -> str:
        return "—" if x is None or x != x else format(x, fmt)

    stamp = datetime.fromtimestamp(snap.ts / 1000, UTC)
    lines = [
        f"=== {snap.symbol} | {stamp:%H:%M:%S} UTC | observed {snap.observed_s:.0f}s"
        " (* = window only partly observed)"
    ]
    lines.append(
        f"  {'flow':<8}{'win':>4}{'chg':>9}{'notional':>11}{'delta':>7}{'tr/s':>7}"
        f"{'int_x':>6}{'vs vwap':>9}{'big buy':>10}{'big sell':>10}"
    )
    for ex, by_window in snap.flow.items():
        for name, w in by_window.items():
            lines.append(
                f"  {ex:<8}{name + ('*' if w.seconds < WINDOWS_S[name] - 1 else ''):>4}"
                f"{f(w.price_change * 100, '+.3f'):>8}%"
                f"{w.notional / 1e3:>10.0f}k{f(w.delta_ratio, '+.2f'):>7}{w.intensity:>7.1f}"
                f"{f(w.intensity_vs_15m, '.2f'):>6}{f(w.price_vs_vwap_bps, '+.1f'):>8}bp"
                f"{w.large_buy_notional / 1e3:>9.0f}k{w.large_sell_notional / 1e3:>9.0f}k"
            )
    for ex, b in snap.book.items():
        depth = " ".join(
            f"±{band}bp {f(bid and bid / 1e3, '.0f')}k/{f(ask and ask / 1e3, '.0f')}k"
            for band, (bid, ask) in b.depth.items()
        )
        slip = " ".join(
            f"${n / 1e3:g}k {f(buy, '.1f')}/{f(sell, '.1f')}"
            for n, (buy, sell) in b.slippage.items()
        )
        lines.append(
            f"  book {ex:<8} spread {b.spread_bps:.2f}bp (x{f(b.spread_vs_15m, '.2f')})  "
            f"seen ±{min(b.visible_bps):.0f}bp  depth bid/ask {depth}  "
            f"imb {f(b.imbalance(10), '+.2f')}  slip buy/sell bp: {slip}"
        )
    c = snap.cross
    lines.append(
        f"  cross basis {f(c.basis_bps, '+.2f')}bp (15m mean {f(c.basis_mean_bps, '+.2f')}, "
        f"std {f(c.basis_std_bps, '.2f')})  {snap.reference} share "
        + " ".join(f"{w}:{f(v, '.0%')}" for w, v in c.reference_volume_share.items())
        + "  confirms "
        + " ".join(
            f"{w}:{'—' if v is None else ('yes' if v else 'NO')}" for w, v in c.confirms.items()
        )
    )
    return "\n".join(lines)


def _risk_manager(settings: Settings) -> RiskManager:
    r = settings.risk
    limits = RiskLimits(
        risk_per_trade_pct=r.risk_per_trade_pct,
        max_leverage=r.max_leverage,
        max_open_positions=r.max_open_positions,
        daily_loss_limit_pct=r.daily_loss_limit_pct,
        max_drawdown_pct=r.max_drawdown_pct,
        max_trades_per_hour=r.max_trades_per_hour,
    )
    return RiskManager(limits, settings.scanner.taker_fee_bps / 1e4, RiskStateStore(r.state_path))


def _risk(settings: Settings, reset: bool) -> None:
    rm = _risk_manager(settings)
    if reset:
        rm.reset_halt()
        print("kill switch cleared; peak equity will restart from current equity")
    state, limits = rm.state, rm.limits
    print(f"limits      : {limits}")
    print(f"kill switch : {state.halted or 'off'}")
    print(f"peak equity : {state.peak_equity:,.2f} USDT")
    print(f"day {state.day or '—'}: start equity {state.day_start_equity:,.2f} USDT")
    print(f"opens/hour  : {len(state.opens)} recorded")


async def _agent(settings: Settings, dry_run: bool, max_ticks: int | None) -> None:
    _cancel_on_shutdown_signals()
    from enduro.agent.claude import ClaudeClient
    from enduro.agent.claude_code import ClaudeCodeBackend
    from enduro.agent.prompt import render_prompt
    from enduro.agent.runtime import AgentConfig, AgentRuntime
    from enduro.journal.journal import Journal
    from enduro.trading.service import TradingService

    market, agent_cfg = settings.market, settings.agent
    if agent_cfg.backend == "api":
        key = settings.anthropic.api_key
        api_key = key.get_secret_value() if key else None
        if not api_key and not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("no Anthropic API key: set ENDURO_ANTHROPIC__API_KEY in .env")
        llm = ClaudeClient(agent_cfg.model, agent_cfg.effort, agent_cfg.max_tokens, api_key)
    else:
        if shutil.which(agent_cfg.claude_bin) is None:
            sys.exit(f"Claude Code CLI not found: {agent_cfg.claude_bin!r}")
        llm = ClaudeCodeBackend(
            agent_cfg.model,
            agent_cfg.effort,
            timeout_s=agent_cfg.claude_timeout_s,
            claude_bin=agent_cfg.claude_bin,
        )

    gateway = _gateway(settings)
    risk = _risk_manager(settings)
    journal = Journal(agent_cfg.journal_dir)
    reference = CcxtSource(market.reference_exchange, market_type=market.market_type)
    execution = CcxtSource(market.execution_exchange, market_type=market.market_type)
    stream_sources = [
        CcxtSource(ex, market_type=market.market_type, book_limit=FOCUS_BOOK_LIMITS.get(ex))
        for ex in (market.reference_exchange, market.execution_exchange)
    ]
    try:
        state = await gateway.account_state()
        if state.margin_mode != "cross" or not state.hedge_mode:
            sys.exit("account is not in cross margin + hedge mode; run `enduro account --setup`")
        symbols = [e.symbol for e in await _build_universe(settings, reference, execution)]
        radar = RadarRunner(
            reference,
            symbols,
            settings.storage.root,
            settings.scanner.history_days,
            settings.scanner.taker_fee_bps,
        )
        log.info("preparing radar for %d symbols", len(symbols))
        await radar.prepare()

        bus = EventBus()
        events = bus.subscribe()
        tracker = FocusTracker(market.reference_exchange, market.execution_exchange)
        collector = Collector(stream_sources, [], bus, book_depth=FOCUS_BOOK_DEPTH)

        async def quote(symbol: str) -> tuple[float, float]:
            """Best bid/ask on the execution exchange: live book if fresh, else REST."""
            book = tracker.latest_book(market.execution_exchange, symbol)
            if book is None or now_ms() - book.recv_ts > 3_000 or book.mid is None:
                book = await execution.fetch_top_of_book(symbol)
            return book.best_bid, book.best_ask

        trading = TradingService(gateway, risk, journal, quote, dry_run=dry_run)
        prompt = render_prompt(
            agent_cfg.prompt_path,
            risk.limits,
            settings.scanner.taker_fee_bps,
            settings.execution.environment,
        )
        runtime = AgentRuntime(
            llm=llm,
            system_prompt=prompt,
            trading=trading,
            radar=radar,
            focus_collector=collector,
            focus_tracker=tracker,
            reference_source=reference,
            journal=journal,
            config=AgentConfig(
                search_interval_s=agent_cfg.search_interval_s,
                focus_interval_s=agent_cfg.focus_interval_s,
                max_llm_calls_per_tick=agent_cfg.max_llm_calls_per_tick,
                max_tool_calls_per_tick=agent_cfg.max_tool_calls_per_tick,
                wake_move_bps=agent_cfg.wake_move_bps,
            ),
            universe=symbols,
            reference=market.reference_exchange,
            execution=market.execution_exchange,
            focus_events=events,
        )
        open_positions = await gateway.positions()
        if open_positions:
            held = open_positions[0].symbol
            runtime.universe.add(held)
            await runtime.set_focus(held, "position already open at start: manage it")
            log.warning("open position on %s at start: focusing on it", held)
        journal.write(
            "start",
            backend=agent_cfg.backend,
            model=agent_cfg.model,
            effort=agent_cfg.effort,
            dry_run=dry_run,
            environment=settings.execution.environment,
            universe=len(symbols),
        )
        log.info(
            "agent started: %s (%s), %s%s",
            agent_cfg.model,
            agent_cfg.effort,
            settings.execution.environment,
            " DRY RUN" if dry_run else "",
        )
        await runtime.run(max_ticks)
        log.info(
            "agent stopped after %d ticks, cost $%.3f", runtime.tick_no, runtime.session_cost_usd
        )
    finally:
        await asyncio.gather(
            gateway.close(),
            reference.close(),
            execution.close(),
            llm.close(),
            return_exceptions=True,
        )


def _feedback(settings: Settings, days: int) -> None:
    from datetime import timedelta

    from enduro.journal.journal import Journal

    journal = Journal(settings.agent.journal_dir)
    today = datetime.now(UTC)
    records = [
        r
        for d in range(days - 1, -1, -1)
        for r in journal.read(f"{today - timedelta(days=d):%Y-%m-%d}")
        if r["kind"] == "feedback"
    ]
    if not records:
        print("no tooling feedback from the agent yet")
        return
    for r in records:
        stamp = datetime.fromtimestamp(r["ts"] / 1000, UTC)
        print(f"{stamp:%m-%d %H:%M} [{r['category']}] {r['title']}  ({r.get('focus') or 'search'})")
        print(f"    {r['details']}")
        if r.get("impact"):
            print(f"    impact: {r['impact']}")


async def _universe(settings: Settings) -> None:
    market = settings.market
    reference = CcxtSource(market.reference_exchange, market_type=market.market_type)
    execution = CcxtSource(market.execution_exchange, market_type=market.market_type)
    try:
        entries = await _build_universe(settings, reference, execution)
    finally:
        await asyncio.gather(reference.close(), execution.close())
    for i, e in enumerate(entries, 1):
        print(f"{i:>3}  {e.symbol:<22} ${e.quote_volume_24h / 1e6:>10.1f}M")
    print(
        f"{len(entries)} symbols on {market.reference_exchange} ∩ {market.execution_exchange} "
        f"with 24h volume >= ${settings.scanner.min_quote_volume_usd / 1e6:.0f}M"
    )


async def _backfill(settings: Settings, days: int) -> None:
    market, root = settings.market, settings.storage.root
    exchanges = [market.reference_exchange, market.execution_exchange]
    sources = [CcxtSource(ex, market_type=market.market_type) for ex in exchanges]
    try:
        entries = await _build_universe(settings, sources[0], sources[1])
        symbols = [e.symbol for e in entries]
        now = now_ms()
        start = now - days * 86_400_000
        log.info("backfilling %d days of 1m candles for %d symbols", days, len(symbols))

        async def run(source: CcxtSource) -> None:
            done = 0

            async def persist(symbol: str, candles: list) -> None:
                nonlocal done
                await asyncio.to_thread(write_candles, root, candles)
                done += 1
                if done % 10 == 0:
                    log.info("%s: %d symbols updated", source.exchange, done)

            last = await asyncio.to_thread(last_candle_ts, root, source.exchange)
            total = await backfill(source, symbols, last, start, now, persist)
            log.info("%s: done, %d candles written", source.exchange, total)

        await asyncio.gather(*(run(src) for src in sources))
    finally:
        await asyncio.gather(*(src.close() for src in sources))


async def _scan(settings: Settings, top: int, as_json: bool, once: bool) -> None:
    market = settings.market
    reference = CcxtSource(market.reference_exchange, market_type=market.market_type)
    execution = CcxtSource(market.execution_exchange, market_type=market.market_type)
    runner: RadarRunner | None = None
    try:
        symbols = [e.symbol for e in await _build_universe(settings, reference, execution)]
        runner = RadarRunner(
            reference,
            symbols,
            settings.storage.root,
            settings.scanner.history_days,
            settings.scanner.taker_fee_bps,
        )
        await runner.prepare()

        def show(rows: list[RadarRow]) -> None:
            if as_json:
                print(
                    json.dumps([r.to_summary() for r in rows[:top]], ensure_ascii=False), flush=True
                )
            else:
                print(_render_radar(rows[:top], reference.exchange, len(symbols)), flush=True)

        if once:
            show(await runner.refresh())
        else:
            await runner.run_forever(show)
    finally:
        if runner is not None:
            runner.flush()
        await asyncio.gather(reference.close(), execution.close())


def _render_radar(rows: list[RadarRow], exchange: str, universe_size: int) -> str:
    def f(x: float, fmt: str) -> str:
        return "—" if x != x else format(x, fmt)  # NaN-safe

    stamp = datetime.fromtimestamp((rows[0].ts if rows else now_ms()) / 1000 + 60, UTC)
    lines = [
        f"=== {stamp:%H:%M} UTC | radar on {exchange} | {universe_size} symbols | "
        "sorted by score = sqrt(vol_x * vlm_x) over 15m vs usual",
        "    usual = norm for this hour over weeks; 24h = vs the last day; "
        "day = 24h volume vs usual (in play?)",
        f"{'#':>2} {'symbol':<14}{'price':>11}{'score':>7}{'day':>6} │"
        f"{'15m chg':>9}{'move':>7}{'vol_x':>6}{'vlm_x':>6}{'v/24h':>6}{'eff':>6} │"
        f"{'1h chg':>8}{'eff':>6} │{'exp':>6}{'atr5m':>7}{'mv/fee':>7}",
    ]
    for i, r in enumerate(rows, 1):
        a, b = r.windows["15m"], r.windows["1h"]
        lines.append(
            f"{i:>2} {r.symbol.split('/')[0]:<14}{r.price:>11.6g}{f(r.score, '.1f'):>7}"
            f"{f(r.day_volume_ratio, '.1f'):>6} │"
            f"{f(a.change * 100, '+.2f'):>8}%{f(a.expected_move * 100, '.2f'):>6}%"
            f"{f(a.vol_ratio, '.1f'):>6}{f(a.volume_ratio, '.1f'):>6}"
            f"{f(a.volume_vs_24h, '.1f'):>6}{f(a.efficiency, '.2f'):>6} │"
            f"{f(b.change * 100, '+.2f'):>7}%{f(b.efficiency, '.2f'):>6} │"
            f"{f(r.expansion, '.2f'):>6}{f(r.atr_5m * 100, '.2f'):>6}%{f(r.move_vs_fees, '.1f'):>7}"
        )
    return "\n".join(lines)


def _render(state: MarketState, settings: Settings, dropped: int, sink: ParquetSink | None) -> str:
    market = settings.market
    ts = now_ms()
    recorded = f"rows written: {sink.rows_written}" if sink else "recording off"
    lines = [f"--- window {state.window_ms // 1000}s | dropped events: {dropped} | {recorded}"]
    for symbol in market.symbols:
        lines.append(symbol)
        for ex in market.exchanges:
            book = state.book(ex, symbol)
            stats = state.trade_stats(ex, symbol, ts)
            mid = f"{book.mid:>12.4f}" if book and book.mid else f"{'—':>12}"
            spread = f"{book.spread_bps:5.2f}" if book and book.spread_bps is not None else "  —  "
            lag = f"{ts - book.recv_ts:>5}ms" if book else "     —"
            buy = f"{stats.buy_ratio:4.0%}" if stats.buy_ratio is not None else "  — "
            lines.append(
                f"  {ex:<8} mid {mid}  spread {spread}bps  age {lag}  "
                f"trades {stats.count:>6}  vol ${stats.notional / 1e6:8.2f}M  buy {buy}"
            )
        for ex in market.exchanges:
            if ex == market.reference_exchange:
                continue
            div = state.divergence_bps(symbol, market.reference_exchange, ex)
            if div is not None:
                lines.append(f"  {ex} vs {market.reference_exchange}: {div:+.2f} bps")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="enduro")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="stream live market data and print a summary")
    collect.add_argument("--symbols", nargs="+", help="override symbols from config")
    collect.add_argument("--interval", type=float, default=5.0, help="report interval, seconds")
    collect.add_argument("--no-record", action="store_true", help="do not write data to disk")

    commands.add_parser("universe", help="list symbols the radar scans")

    test_trade = commands.add_parser(
        "test-trade", help="demo only: open the minimum size at market and close it"
    )
    test_trade.add_argument("symbol", help="e.g. BTC/USDT:USDT")
    test_trade.add_argument("--side", choices=["long", "short"], default="long")
    test_trade.add_argument(
        "--with-stop", type=float, metavar="PCT", help="attach a stop loss PCT%% away"
    )

    risk = commands.add_parser("risk", help="show risk limits and state")
    risk.add_argument("--reset", action="store_true", help="manually clear the kill switch")

    account = commands.add_parser("account", help="show execution account state")
    account.add_argument(
        "--setup", action="store_true", help="switch the account to cross margin + hedge mode"
    )

    fill = commands.add_parser("backfill", help="download/update 1m candle history")
    fill.add_argument("--days", type=int, help="history depth (default: scanner.history_days)")

    scan = commands.add_parser("scan", help="rank the market by unusual activity, every minute")
    scan.add_argument("--top", type=int, default=15, help="rows to show")
    scan.add_argument("--json", action="store_true", help="print agent-facing JSON")
    scan.add_argument("--once", action="store_true", help="scan once and exit")

    focus = commands.add_parser("focus", help="live microstructure metrics for chosen symbols")
    focus.add_argument("symbols", nargs="+", help="e.g. ETH/USDT:USDT")
    focus.add_argument("--interval", type=float, default=10.0, help="report interval, seconds")
    focus.add_argument("--json", action="store_true", help="print agent-facing JSON")

    agent = commands.add_parser("agent", help="run the LLM trading agent")
    agent.add_argument(
        "--dry-run", action="store_true", help="decide and journal, but never send orders"
    )
    agent.add_argument("--ticks", type=int, help="stop after N ticks")

    feedback = commands.add_parser("feedback", help="tooling gaps reported by the agent")
    feedback.add_argument("--days", type=int, default=7, help="how many days back")

    sql = commands.add_parser("sql", help="query recorded data (views: trades, books)")
    sql.add_argument("query", help='e.g. "select exchange, count(*) from trades group by 1"')

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    settings = Settings()
    if args.command == "collect":
        if args.symbols:
            settings.market.symbols = args.symbols
        if args.no_record:
            settings.storage.enabled = False
        with contextlib.suppress(asyncio.CancelledError):
            asyncio.run(_collect(settings, args.interval))
    elif args.command == "account":
        asyncio.run(_account(settings, args.setup))
    elif args.command == "test-trade":
        asyncio.run(_test_trade(settings, args.symbol, args.side, args.with_stop))
    elif args.command == "risk":
        _risk(settings, args.reset)
    elif args.command == "universe":
        asyncio.run(_universe(settings))
    elif args.command == "backfill":
        asyncio.run(_backfill(settings, args.days or settings.scanner.history_days))
    elif args.command == "scan":
        with contextlib.suppress(asyncio.CancelledError):
            asyncio.run(_scan_with_signals(settings, args.top, args.json, args.once))
    elif args.command == "focus":
        with contextlib.suppress(asyncio.CancelledError):
            asyncio.run(_focus(settings, args.symbols, args.interval, args.json))
    elif args.command == "agent":
        with contextlib.suppress(asyncio.CancelledError):
            asyncio.run(_agent(settings, args.dry_run, args.ticks))
    elif args.command == "feedback":
        _feedback(settings, args.days)
    elif args.command == "sql":
        try:
            store.connect(settings.storage.root).sql(args.query).show(max_rows=100)
        except Exception as e:  # surface DuckDB errors as a clean CLI message
            sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
