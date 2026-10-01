"""Price alerts the agent leaves on levels it cares about.

The agent often releases a coin with a plan ("short if a 1m candle closes below 0.5225")
and then nobody watches the level. An alert keeps that plan alive cheaply: it is checked
against each closed 1m candle of the reference exchange (the radar's feed, so any coin in
the universe works, focused or not) and wakes the agent once when a candle closes beyond
the level. What to do then is the agent's call — an alert never trades.

Alerts are kept in a small JSON file so they survive agent restarts; expired ones are
dropped on load.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from enduro.core.models import MINUTE_MS, Candle

log = logging.getLogger(__name__)

DIRECTIONS = ("above", "below")
MAX_ALERTS = 10
DEFAULT_TTL_MIN = 120
MAX_TTL_MIN = 480


@dataclass(slots=True)
class Alert:
    id: int
    symbol: str
    level: float
    direction: str  # "above" | "below": a 1m close beyond the level fires it
    note: str
    created_ms: int
    expires_ms: int
    checked_ts: int = 0  # open time of the last candle already evaluated

    def crossed(self, close: float) -> bool:
        return close > self.level if self.direction == "above" else close < self.level

    def to_summary(self, now_ms: int) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "fires_on_1m_close": f"{self.direction} {self.level:g}",
            "note": self.note,
            "expires_in_min": max(0, round((self.expires_ms - now_ms) / MINUTE_MS)),
        }


@dataclass(frozen=True, slots=True)
class FiredAlert:
    alert: Alert
    candle: Candle

    def describe(self) -> str:
        a, c = self.alert, self.candle
        return (
            f"alert #{a.id} {a.symbol}: 1m close {c.close:g} {a.direction} {a.level:g}"
            f" (your note: {a.note})"
        )


class AlertBook:
    def __init__(self, max_alerts: int = MAX_ALERTS, path: Path | str | None = None) -> None:
        self.max_alerts = max_alerts
        self.path = Path(path) if path else None
        self._alerts: dict[int, Alert] = {}
        self._next_id = 1
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            for fields in saved["alerts"]:
                alert = Alert(**fields)
                self._alerts[alert.id] = alert
            self._next_id = max(saved.get("next_id", 1), max(self._alerts, default=0) + 1)
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("cannot read saved alerts from %s; starting empty", self.path)
            self._alerts.clear()

    def _save(self) -> None:
        if self.path is None:
            return
        data = {"next_id": self._next_id, "alerts": [asdict(a) for a in self._alerts.values()]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(
        self, symbol: str, level: float, direction: str, note: str, ttl_min: int, now_ms: int
    ) -> Alert:
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if not level > 0:
            raise ValueError("level must be a positive price")
        if not 1 <= ttl_min <= MAX_TTL_MIN:
            raise ValueError(f"expires_minutes must be 1..{MAX_TTL_MIN}")
        self.prune(now_ms)
        if len(self._alerts) >= self.max_alerts:
            raise ValueError(f"at most {self.max_alerts} alerts: cancel one first")
        alert = Alert(
            self._next_id, symbol, level, direction, note, now_ms, now_ms + ttl_min * MINUTE_MS
        )
        self._alerts[alert.id] = alert
        self._next_id += 1
        self._save()
        return alert

    def cancel(self, alert_id: int) -> Alert | None:
        alert = self._alerts.pop(alert_id, None)
        if alert is not None:
            self._save()
        return alert

    def prune(self, now_ms: int) -> list[Alert]:
        """Drop expired alerts and return them."""
        expired = [a for a in self._alerts.values() if a.expires_ms <= now_ms]
        for a in expired:
            del self._alerts[a.id]
        if expired:
            self._save()
        return expired

    def active(self, now_ms: int) -> list[Alert]:
        self.prune(now_ms)
        return list(self._alerts.values())

    def check(self, candles: Callable[[str], Iterable[Candle]], now_ms: int) -> list[FiredAlert]:
        """Evaluate candles closed since each alert was set; fired alerts are removed.

        `candles(symbol)` returns the latest closed 1m candles of a symbol, oldest first."""
        fired = []
        for alert in self.active(now_ms):
            for c in candles(alert.symbol):
                # Only candles that closed after the alert was set, each one once.
                if c.ts <= alert.checked_ts or c.ts + MINUTE_MS <= alert.created_ms:
                    continue
                alert.checked_ts = c.ts
                if alert.crossed(c.close):
                    fired.append(FiredAlert(alert, c))
                    del self._alerts[alert.id]
                    break
        if fired:
            self._save()
        return fired
