"""Tools the trading agent can call. Same set in both modes; handlers enforce the mode.

Tool results are compact JSON. Errors go back as `is_error` results with a plain-language
reason, so the model can correct itself instead of the tick failing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from enduro.agent.llm import ToolSpec
from enduro.risk.manager import OpenIntent

if TYPE_CHECKING:
    from enduro.agent.runtime import AgentRuntime

MIN_CHECK_S, MAX_CHECK_S = 15, 600
HISTORY_INTERVALS = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
MAX_HISTORY_BARS = 120


class ToolInputError(ValueError):
    """Bad tool input: reported back to the model as an error result."""


@dataclass(frozen=True, slots=True)
class Tool:
    spec: ToolSpec
    handler: Callable[[AgentRuntime, dict[str, Any]], Awaitable[dict[str, Any]]]


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _num(args: dict[str, Any], key: str, required: bool = True) -> float | None:
    value = args.get(key)
    if value is None:
        if required:
            raise ToolInputError(f"'{key}' is required")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ToolInputError(f"'{key}' must be a number") from None
    if not math.isfinite(number) or number < 0:
        raise ToolInputError(f"'{key}' must be a non-negative finite number")
    return number


def _side(args: dict[str, Any]) -> str:
    side = args.get("side")
    if side not in ("long", "short"):
        raise ToolInputError("'side' must be 'long' or 'short'")
    return side


def _text(args: dict[str, Any], key: str) -> str:
    value = str(args.get(key) or "").strip()
    if not value:
        raise ToolInputError(f"'{key}' is required: explain it in a sentence or two")
    return value


# ---------------------------------------------------------------- handlers


async def get_radar(rt: AgentRuntime, args: dict[str, Any]) -> dict[str, Any]:
    top = int(args.get("top") or 10)
    return rt.radar_view(max(1, min(top, 25)))


async def get_focus(rt: AgentRuntime, args: dict[str, Any]) -> dict[str, Any]:
    if rt.focus_symbol is None:
        raise ToolInputError("no focus symbol; call set_focus first")
    return rt.focus_view()


async def get_price_history(rt: AgentRuntime, args: dict[str, Any]) -> dict[str, Any]:
    symbol = str(args.get("symbol") or "")
    interval = args.get("interval")
    if interval not in HISTORY_INTERVALS:
        raise ToolInputError(f"'interval' must be one of {list(HISTORY_INTERVALS)}")
    bars = int(args.get("bars") or 60)
    if not 1 <= bars <= MAX_HISTORY_BARS:
        raise ToolInputError(f"'bars' must be between 1 and {MAX_HISTORY_BARS}")
    rt.require_known_symbol(symbol)
    candles = await rt.price_history(symbol, interval, bars)
    return {
        "symbol": symbol,
        "exchange": rt.reference,
        "interval": interval,
        "columns": ["time_utc", "open", "high", "low", "close", "volume_usd"],
        "bars": [
            [
                f"{datetime.fromtimestamp(c.ts / 1000, UTC):%m-%d %H:%M}",
                c.open,
                c.high,
                c.low,
                c.close,
                round(c.close * c.volume),
            ]
            for c in candles
        ],
    }


async def set_focus(rt: AgentRuntime, args: dict[str, Any]) -> dict[str, Any]:
    symbol = str(args.get("symbol") or "")
    reason = _text(args, "reason")
    rt.require_known_symbol(symbol)
    await rt.set_focus(symbol, reason)
    return {
        "focus": symbol,
        "note": "live trades and order books are streaming now; flow metrics need a few "
        "minutes to fill — use get_price_history meanwhile",
    }


async def release_focus(rt: AgentRuntime, args: dict[str, Any]) -> dict[str, Any]:
    reason = _text(args, "reason")
    if rt.focus_symbol is None:
        raise ToolInputError("there is no focus to release")
    if await rt.has_open_position(rt.focus_symbol):
        raise ToolInputError("close the open position on the focus symbol before releasing it")
    released = rt.focus_symbol
    await rt.release_focus(reason)
    return {"released": released, "mode": "search"}


async def get_account(rt: AgentRuntime, args: dict[str, Any]) -> dict[str, Any]:
    return await rt.trading.account()


async def open_position(rt: AgentRuntime, args: dict[str, Any]) -> dict[str, Any]:
    if rt.focus_symbol is None:
        raise ToolInputError("trades are only allowed on the focus symbol; set_focus first")
    risk_pct = _num(args, "risk_pct", required=False)
    intent = OpenIntent(
        symbol=rt.focus_symbol,
        side=_side(args),
        stop_loss=_num(args, "stop_loss"),
        take_profit=_num(args, "take_profit", required=False) or None,
        risk_pct=risk_pct,
    )
    return await rt.trading.open(intent, thesis=_text(args, "thesis"))


async def close_position(rt: AgentRuntime, args: dict[str, Any]) -> dict[str, Any]:
    if rt.focus_symbol is None:
        raise ToolInputError("no focus symbol")
    return await rt.trading.close(rt.focus_symbol, _side(args), reason=_text(args, "reason"))


async def update_protection(rt: AgentRuntime, args: dict[str, Any]) -> dict[str, Any]:
    if rt.focus_symbol is None:
        raise ToolInputError("no focus symbol")
    stop = _num(args, "stop_loss", required=False)
    take = _num(args, "take_profit", required=False)
    if stop is None and take is None:
        raise ToolInputError("give stop_loss and/or take_profit")
    return await rt.trading.protect(rt.focus_symbol, _side(args), stop, take)


FEEDBACK_CATEGORIES = [
    "missing_data",
    "missing_tool",
    "tool_problem",
    "execution",
    "risk_limit",
    "other",
]


async def report_tooling_gap(rt: AgentRuntime, args: dict[str, Any]) -> dict[str, Any]:
    category = args.get("category")
    if category not in FEEDBACK_CATEGORIES:
        raise ToolInputError(f"'category' must be one of {FEEDBACK_CATEGORIES}")
    rt.journal.write(
        "feedback",
        n=rt.tick_no,
        focus=rt.focus_symbol,
        category=category,
        title=_text(args, "title"),
        details=_text(args, "details"),
        impact=str(args.get("impact") or "").strip(),
    )
    return {"recorded": True, "note": "thanks — the developers read these to improve your tools"}


async def finish_tick(rt: AgentRuntime, args: dict[str, Any]) -> dict[str, Any]:
    seconds = int(_num(args, "next_check_seconds") or 0)
    if not MIN_CHECK_S <= seconds <= MAX_CHECK_S:
        raise ToolInputError(f"'next_check_seconds' must be {MIN_CHECK_S}..{MAX_CHECK_S}")
    rt.finish_tick(seconds, _text(args, "note"))
    return {"ok": True}


# ---------------------------------------------------------------- registry

_SIDE = {"type": "string", "enum": ["long", "short"]}

TOOLS: list[Tool] = [
    Tool(
        ToolSpec(
            "get_radar",
            "Rank the scanned universe by how unusually active each coin is right now "
            "(1m candles on the reference exchange vs each coin's own norms). Use it to "
            "find a coin worth focusing on.",
            _schema({"top": {"type": "integer", "minimum": 1, "maximum": 25}}, []),
        ),
        get_radar,
    ),
    Tool(
        ToolSpec(
            "get_focus",
            "Live microstructure of the focus coin on both exchanges: taker flow and "
            "delta, intensity, large prints, VWAP, spread, depth, slippage, and whether "
            "the reference exchange confirms the move.",
            _schema({}, []),
        ),
        get_focus,
    ),
    Tool(
        ToolSpec(
            "get_price_history",
            "OHLCV bars from the reference exchange for structure and levels.",
            _schema(
                {
                    "symbol": {"type": "string"},
                    "interval": {"type": "string", "enum": list(HISTORY_INTERVALS)},
                    "bars": {"type": "integer", "minimum": 1, "maximum": MAX_HISTORY_BARS},
                },
                ["symbol", "interval"],
            ),
        ),
        get_price_history,
    ),
    Tool(
        ToolSpec(
            "set_focus",
            "Start focusing on one coin from the radar universe: live trades and order "
            "books start streaming and the agent switches to focus mode.",
            _schema(
                {"symbol": {"type": "string"}, "reason": {"type": "string"}},
                ["symbol", "reason"],
            ),
        ),
        set_focus,
    ),
    Tool(
        ToolSpec(
            "release_focus",
            "Stop focusing (only when flat) and return to search mode.",
            _schema({"reason": {"type": "string"}}, ["reason"]),
        ),
        release_focus,
    ),
    Tool(
        ToolSpec(
            "get_account",
            "Equity, open positions with stops, and the risk manager's state and limits.",
            _schema({}, []),
        ),
        get_account,
    ),
    Tool(
        ToolSpec(
            "open_position",
            "Open a position on the focus coin at market. The stop loss is mandatory and "
            "is placed on the exchange with the entry. Size is computed by the risk "
            "manager from the stop distance (risk_pct defaults to the per-trade maximum; "
            "you may risk less). The risk manager can reject the trade and will say why.",
            _schema(
                {
                    "side": _SIDE,
                    "stop_loss": {"type": "number"},
                    "take_profit": {"type": "number"},
                    "risk_pct": {"type": "number", "exclusiveMinimum": 0},
                    "thesis": {"type": "string"},
                },
                ["side", "stop_loss", "thesis"],
            ),
        ),
        open_position,
    ),
    Tool(
        ToolSpec(
            "close_position",
            "Close the whole position on the focus coin at market.",
            _schema({"side": _SIDE, "reason": {"type": "string"}}, ["side", "reason"]),
        ),
        close_position,
    ),
    Tool(
        ToolSpec(
            "update_protection",
            "Move the stop loss and/or take profit of an open position (0 removes the "
            "take profit). A stop can be tightened freely; loosening it is allowed only "
            "within the per-trade risk budget; it can never be removed.",
            _schema(
                {
                    "side": _SIDE,
                    "stop_loss": {"type": "number"},
                    "take_profit": {"type": "number"},
                },
                ["side"],
            ),
        ),
        update_protection,
    ),
    Tool(
        ToolSpec(
            "report_tooling_gap",
            "Tell the developers about a limitation that kept you from trading well or "
            "from making (more) profit: data you needed but could not get, a tool that is "
            "missing or behaves badly, execution problems, a risk limit that blocked a "
            "sound trade. Report each distinct issue once; it does not end the check.",
            _schema(
                {
                    "category": {"type": "string", "enum": FEEDBACK_CATEGORIES},
                    "title": {"type": "string"},
                    "details": {"type": "string"},
                    "impact": {"type": "string"},
                },
                ["category", "title", "details"],
            ),
        ),
        report_tooling_gap,
    ),
    Tool(
        ToolSpec(
            "finish_tick",
            "End this check: when to look again and a short note to your future self "
            "(what you see, what you expect, what would change your mind).",
            _schema(
                {
                    "next_check_seconds": {
                        "type": "integer",
                        "minimum": MIN_CHECK_S,
                        "maximum": MAX_CHECK_S,
                    },
                    "note": {"type": "string"},
                },
                ["next_check_seconds", "note"],
            ),
        ),
        finish_tick,
    ),
]

TOOLS_BY_NAME = {t.spec.name: t for t in TOOLS}


def to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
