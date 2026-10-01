"""Renders the agent's system prompt from prompts/*.md with the live configuration.

The prompt file is plain text with `$placeholders` (string.Template), so it stays
provider-agnostic and easy to edit; numbers always match what the code enforces.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from enduro.risk.manager import RiskLimits

ENVIRONMENT_NOTES = {
    "demo": (
        "Ты торгуешь на демо-счёте: деньги виртуальные, но относись к ним как к настоящим —\n"
        "по результатам этого периода решается, будешь ли ты торговать реальными."
    ),
    "live": "Ты торгуешь реальными деньгами.",
}


def render_prompt(path: Path, limits: RiskLimits, taker_fee_bps: float, environment: str) -> str:
    def fmt(x: float) -> str:
        return f"{x:g}"

    return Template(path.read_text(encoding="utf-8")).substitute(
        risk_per_trade_pct=fmt(limits.risk_per_trade_pct),
        max_leverage=fmt(limits.max_leverage),
        max_open_positions=limits.max_open_positions,
        max_trades_per_hour=limits.max_trades_per_hour,
        daily_loss_limit_pct=fmt(limits.daily_loss_limit_pct),
        max_drawdown_pct=fmt(limits.max_drawdown_pct),
        taker_fee_pct=fmt(taker_fee_bps / 100),
        environment_note=ENVIRONMENT_NOTES[environment],
    )
