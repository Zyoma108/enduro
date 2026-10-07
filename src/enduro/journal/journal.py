"""Append-only decision journal: what the agent saw, decided, why, and what happened.

One JSON object per line, one file per UTC day: state/journal/YYYY-MM-DD.jsonl.
Every record has `ts` (epoch ms), `kind` and kind-specific fields. Kinds in use:
  tick         — an agent tick started (mode, trigger)
  llm          — one model call (model, stop reason, token usage, cost estimate)
  tool         — a tool call made by the agent (name, input, result)
  risk         — a risk decision on an open intent
  order        — an order sent and its outcome
  closed       — a position closed as the exchange books it (net PnL, who closed it)
  note         — the agent's own note at the end of a tick
  focus        — focus set or released, with the reason
  alert        — a price alert set, fired, cancelled or expired
  feedback     — a tooling gap the agent reported
  start / stop — an agent session began / ended
  session      — session-level events (e.g. tick limit reached with a position open)
  error        — anything that went wrong
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from enduro.core.models import now_ms

log = logging.getLogger(__name__)


class Journal:
    def __init__(
        self, root: Path | str, echo: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()
        self._echo = echo  # e.g. print decisions to the console as they are journaled

    def write(self, kind: str, **fields: Any) -> dict[str, Any]:
        record = {"ts": now_ms(), "kind": kind, **fields}
        line = json.dumps(record, ensure_ascii=False, default=_jsonable)
        path = self.root / f"{datetime.fromtimestamp(record['ts'] / 1000, UTC):%Y-%m-%d}.jsonl"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self._echo is not None:
            try:
                self._echo(json.loads(line))  # the record as it reads back from disk
            except Exception:
                log.exception("journal echo failed")
        return record

    def read(self, day: str | None = None) -> list[dict[str, Any]]:
        day = day or f"{datetime.now(UTC):%Y-%m-%d}"
        path = self.root / f"{day}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def recent(self, kind: str, limit: int, since_ms: int = 0) -> list[dict[str, Any]]:
        """Most recent records of `kind` from today and yesterday (and not older than
        `since_ms`), oldest first."""
        today = datetime.now(UTC)
        days = [
            f"{today.fromtimestamp(today.timestamp() - 86_400, UTC):%Y-%m-%d}",
            f"{today:%Y-%m-%d}",
        ]
        records = [
            r for day in days for r in self.read(day) if r["kind"] == kind and r["ts"] >= since_ms
        ]
        return records[-limit:]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)
