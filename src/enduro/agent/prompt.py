"""Renders the agent's system prompt from prompts/*.md with the live configuration.

The prompt file is plain text with `$placeholders` (string.Template), so it stays
provider-agnostic and easy to edit; numbers always match what the code enforces.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from enduro.risk.manager import RiskLimits

POSITIONING_FILE = "positioning.md"  # next to the main prompt

ENVIRONMENT_NOTES = {
    "demo": (
        "Ты торгуешь на демо-счёте: деньги виртуальные, но относись к ним как к настоящим —\n"
        "по результатам этого периода решается, будешь ли ты торговать реальными."
    ),
    "live": "Ты торгуешь реальными деньгами.",
}


def render_prompt(
    path: Path,
    limits: RiskLimits,
    taker_fee_bps: float,
    environment: str,
    min_check_flat_s: int = 120,
    positioning: bool = True,
) -> str:
    """`positioning`: include prompts/positioning.md (open interest and funding) at
    `$positioning` — only when the focus view shows that data."""

    def fmt(x: float) -> str:
        return f"{x:g}"

    fragment = path.with_name(POSITIONING_FILE)
    positioning_text = (
        "\n" + fragment.read_text(encoding="utf-8").rstrip("\n") if positioning else ""
    )

    return Template(path.read_text(encoding="utf-8")).substitute(
        risk_per_trade_pct=fmt(limits.risk_per_trade_pct),
        max_leverage=fmt(limits.max_leverage),
        max_open_positions=limits.max_open_positions,
        max_trades_per_hour=limits.max_trades_per_hour,
        daily_loss_limit_pct=fmt(limits.daily_loss_limit_pct),
        max_drawdown_pct=fmt(limits.max_drawdown_pct),
        taker_fee_pct=fmt(taker_fee_bps / 100),
        environment_note=ENVIRONMENT_NOTES[environment],
        min_check_flat_s=min_check_flat_s,
        positioning=positioning_text,
    )
