"""Command-line entry point: `enduro <command>`."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal

from enduro.analytics.market_state import MarketState
from enduro.config import Settings
from enduro.core.models import now_ms
from enduro.data.bus import EventBus
from enduro.data.ccxt_source import CcxtSource
from enduro.data.collector import Collector


def _cancel_on_shutdown_signals() -> None:
    """Turn SIGINT/SIGTERM into cancellation of the current task for a clean shutdown."""
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)


async def _collect(settings: Settings, interval_s: float) -> None:
    _cancel_on_shutdown_signals()
    market = settings.market
    bus = EventBus()
    queue = bus.subscribe()
    state = MarketState()
    sources = [CcxtSource(ex, market_type=market.market_type) for ex in market.exchanges]
    collector = Collector(sources, market.symbols, bus, book_depth=market.book_depth)

    async def consume() -> None:
        while True:
            state.on_event(await queue.get())

    async def report() -> None:
        while True:
            await asyncio.sleep(interval_s)
            print(_render(state, settings, bus.dropped), flush=True)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(collector.run())
        tg.create_task(consume())
        tg.create_task(report())


def _render(state: MarketState, settings: Settings, dropped: int) -> str:
    market = settings.market
    ts = now_ms()
    lines = [f"--- window {state.window_ms // 1000}s | dropped events: {dropped}"]
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

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    settings = Settings()
    if args.command == "collect":
        if args.symbols:
            settings.market.symbols = args.symbols
        with contextlib.suppress(asyncio.CancelledError):
            asyncio.run(_collect(settings, args.interval))


if __name__ == "__main__":
    main()
