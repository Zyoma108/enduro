"""Human-readable lines for journal records: the agent's console and `enduro journal`.

Decisions (focus, risk, orders, notes, alerts, feedback, errors) are always shown; the
raw machinery (every tool call, every model call) only with `verbose`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

INDENT = "    "


def _num(value: Any) -> str:
    return "—" if value is None else f"{value:g}" if isinstance(value, float) else str(value)


def _short(symbol: str | None) -> str:
    return (symbol or "").split("/")[0] or "search"


def _block(text: str | None) -> str:
    """Multi-line text indented under its header line."""
    lines = (text or "").strip().splitlines() or ["(empty)"]
    return "\n".join(INDENT + line for line in lines)


def _risk(r: dict[str, Any]) -> str:
    if "protection" in r:
        position, new = r.get("position") or {}, r["protection"]
        changes = [
            f"{label} {_num(position.get(key))} → {_num(new[key])}"
            for key, label in (("stop_loss", "stop"), ("take_profit", "take"))
            if new.get(key) is not None
        ]
        head = f"PROTECT {_short(position.get('symbol'))} {position.get('side', '')}: " + ", ".join(
            changes
        )
        return head if r.get("ok") else f"{head} — REJECTED: {r.get('reason')}"
    intent, decision = r.get("intent") or {}, r.get("decision") or {}
    what = (
        f"{intent.get('side')} {_short(intent.get('symbol'))} stop {_num(intent.get('stop_loss'))}"
        f" take {_num(intent.get('take_profit'))}"
    )
    if decision.get("approved"):
        head = (
            f"RISK ok: {what} · qty {_num(decision.get('qty'))}"
            f" @~{_num(decision.get('entry_price'))}"
            f" · risk {decision.get('risk_usd', 0):.2f} USDT ({decision.get('risk_pct', 0):.2f}%)"
        )
    else:
        head = f"RISK REJECTED: {what} · {'; '.join(decision.get('reasons') or [])}"
    return f"{head}\n{_block(r.get('thesis'))}"


def _order(r: dict[str, Any]) -> str:
    action = r.get("action")
    if r.get("dry_run"):
        intent = r.get("intent") or {}
        symbol = intent.get("symbol") or r.get("symbol")
        side = intent.get("side") or r.get("side")
        return f"ORDER (dry run) {action} {side} {_short(symbol)}"
    request, result = r.get("request") or {}, r.get("result") or {}
    head = (
        f"ORDER {action} {request.get('position_side')} {_short(request.get('symbol'))}"
        f" {_num(result.get('filled'))} @ {_num(result.get('avg_price'))}"
        f" · fee {result.get('fee') or 0:.4f}"
    )
    if r.get("gross_pnl_usdt") is not None:
        head += f" · gross PnL {r['gross_pnl_usdt']:+.2f} (entry {_num(r.get('entry_price'))})"
    if result.get("status") not in (None, "closed"):
        head += f" · status {result.get('status')}"
    return f"{head}\n{_block(r['reason'])}" if r.get("reason") else head


def _alert(r: dict[str, Any]) -> str:
    a = r.get("alert") or {}
    head = f"ALERT #{a.get('id')} {r.get('action')}: {_short(a.get('symbol'))}"
    rule = f"1m close {a.get('direction')} {_num(a.get('level'))}"
    if r.get("action") == "fired":
        return f"{head} closed {_num(r.get('close'))}, {a.get('direction')} {_num(a.get('level'))}"
    if r.get("action") == "set":
        minutes = round((a.get("expires_ms", 0) - a.get("created_ms", 0)) / 60_000)
        return f"{head} on {rule} for {minutes} min\n{_block(a.get('note'))}"
    return f"{head} ({rule})"


def _compact(value: Any, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_record(r: dict[str, Any], verbose: bool = False) -> str | None:
    """One record as text (may span lines), or None when it is hidden at this verbosity."""
    kind = r.get("kind")
    if kind == "start":
        dry = " · DRY RUN" if r.get("dry_run") else ""
        return (
            f"START {r.get('backend')} · {r.get('model')} ({r.get('effort')}) · "
            f"{r.get('environment')}{dry} · universe {r.get('universe')}"
        )
    if kind == "stop":
        return f"STOP after {r.get('ticks')} ticks · session cost ${r.get('cost_usd', 0):.3f}"
    if kind == "tick":
        return f"TICK {r.get('n')} · {_short(r.get('focus'))} · {r.get('trigger')}"
    if kind == "focus":
        if r.get("symbol"):
            return f"FOCUS {_short(r['symbol'])}\n{_block(r.get('reason'))}"
        return f"RELEASE {_short(r.get('released'))}\n{_block(r.get('reason'))}"
    if kind == "risk":
        return _risk(r)
    if kind == "order":
        return _order(r)
    if kind == "alert":
        return _alert(r)
    if kind == "note":
        meta = [f"tick {r.get('n')}", _short(r.get("focus"))]
        if r.get("next_check_s") is not None:
            meta.append(f"next in {r['next_check_s']}s")
        if r.get("session_cost_usd") is not None:
            meta.append(f"${r['session_cost_usd']:.3f}")
        return f"NOTE {' · '.join(meta)}\n{_block(r.get('text'))}"
    if kind == "feedback":
        text = f"FEEDBACK [{r.get('category')}] {r.get('title')}\n{_block(r.get('details'))}"
        return text + (f"\n{INDENT}impact: {r['impact']}" if r.get("impact") else "")
    if kind == "error":
        return f"ERROR {r.get('what', '')}: {r.get('error')}"
    if not verbose:
        return None
    if kind == "tool":
        outcome = f"error: {r['error']}" if "error" in r else _compact(r.get("result"))
        return f"{INDENT}tool {r.get('name')} {_compact(r.get('input'))} → {outcome}"
    if kind == "llm":
        usage = r.get("usage") or {}
        return (
            f"{INDENT}llm {r.get('model')} {r.get('stop_reason')} · "
            f"out {usage.get('output_tokens')} tok · ${usage.get('cost_usd') or 0:.3f}"
        )
    return f"{INDENT}{kind} {_compact({k: v for k, v in r.items() if k not in ('ts', 'kind')})}"


def format_line(r: dict[str, Any], verbose: bool = False) -> str | None:
    """`format_record` prefixed with the record's UTC time."""
    text = format_record(r, verbose)
    if text is None:
        return None
    return f"{datetime.fromtimestamp(r['ts'] / 1000, UTC):%H:%M:%S} {text}"


def day_path(root: Path | str, day: str) -> Path:
    return Path(root) / f"{day}.jsonl"


def follow(root: Path | str, day: str, poll_s: float = 1.0) -> Iterator[dict[str, Any]]:
    """Records of `day` from the start, then new ones as they are written; rolls over to
    the next UTC day's file at midnight. Never returns."""
    position = 0
    while True:
        path = day_path(root, day)
        if path.exists():
            with path.open(encoding="utf-8") as f:
                f.seek(position)
                while line := f.readline():
                    if not line.endswith("\n"):  # half-written: read it whole next time
                        break
                    position = f.tell()
                    yield json.loads(line)
        today = f"{datetime.now(UTC):%Y-%m-%d}"
        if today > day and day_path(root, today).exists():
            day, position = today, 0
            continue
        time.sleep(poll_s)
