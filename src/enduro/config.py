"""Application settings.

Priority (highest first): constructor args → env vars (ENDURO_*) → .env → config.toml.
Secrets must only come from env / .env, never from config.toml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class MarketConfig(BaseModel):
    exchanges: list[str] = Field(default_factory=lambda: ["binance", "bybit"])
    reference_exchange: str = "binance"
    execution_exchange: str = "bybit"
    market_type: Literal["spot", "swap"] = "swap"
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT:USDT"])
    book_depth: int = Field(default=20, ge=1, le=200)

    @model_validator(mode="after")
    def _roles_are_known_exchanges(self) -> Self:
        for role in ("reference_exchange", "execution_exchange"):
            if getattr(self, role) not in self.exchanges:
                raise ValueError(f"{role}={getattr(self, role)!r} is not in exchanges")
        return self


class StorageConfig(BaseModel):
    enabled: bool = True
    root: Path = Path("data")
    flush_interval_s: float = Field(default=60.0, gt=0)
    book_snapshot_interval_ms: int = Field(default=1_000, ge=0)


class ScannerConfig(BaseModel):
    # Minimum 24h volume on the execution exchange (USDT) for a symbol to enter the radar.
    min_quote_volume_usd: float = Field(default=10_000_000, ge=0)
    # "crypto" and/or "tradfi" (tokenized stocks, ETFs, commodities, FX, pre-IPO).
    asset_classes: list[Literal["crypto", "tradfi"]] = Field(default_factory=lambda: ["crypto"])
    # Candle history used to compute "normal" volatility and volume per hour of day.
    history_days: int = Field(default=28, ge=1)
    # Tradable on the execution exchange: resting depth within ±10 bps of mid on the thinner
    # side (USDT) and spread, judged on the median of the last few book snapshots.
    min_depth_10bps_usd: float = Field(default=2_000.0, ge=0)
    max_spread_bps: float = Field(default=10.0, gt=0)
    # Taker fee on the execution exchange, bps per side (Bybit VIP0 perps: 5.5).
    taker_fee_bps: float = Field(default=5.5, ge=0)


class ExecutionConfig(BaseModel):
    # "demo" — Bybit Demo Trading (virtual money); "live" — real money.
    environment: Literal["demo", "live"] = "demo"
    # Second, independent switch: live trading is refused unless this is true.
    allow_live: bool = False
    # Entries and manual closes work a post-only limit order (maker fee) at the touch.
    open_chase_s: float = Field(default=20.0, gt=0)
    open_chase_max_stop_share: float = Field(default=0.1, ge=0, le=0.5)
    close_chase_s: float = Field(default=15.0, ge=0)
    chase_poll_s: float = Field(default=1.0, gt=0)


class RiskConfig(BaseModel):
    """Hard limits enforced in code; the agent cannot override them."""

    risk_per_trade_pct: float = Field(default=1.0, gt=0, le=5)
    max_leverage: float = Field(default=5.0, gt=0, le=20)
    max_open_positions: int = Field(default=1, ge=1)
    daily_loss_limit_pct: float = Field(default=5.0, gt=0)
    max_drawdown_pct: float = Field(default=10.0, gt=0)
    max_trades_per_hour: int = Field(default=6, ge=1)
    # Peak equity, start-of-day equity, kill switch — must survive restarts.
    state_path: Path = Path("state/risk.json")


Credentials = Literal["anthropic", "deepseek", "glm"]


class ModelProfile(BaseModel):
    """One way to run the trader's model: `[models.<name>]` in config.toml.

    backend:
      "claude-code" — Claude Code CLI (`claude -p`) on the account it is logged into;
      "anthropic"   — Anthropic Messages API;
      "openai"      — any OpenAI-compatible Chat Completions API (DeepSeek, Z.ai GLM).
    """

    backend: Literal["claude-code", "anthropic", "openai"]
    model: str
    # Claude: low | medium | high | xhigh | max. OpenAI-compatible: sent as
    # `reasoning_effort` when set (DeepSeek: low | high | max).
    effort: str | None = None
    base_url: str | None = None  # required for "openai"
    # Which ENDURO_<NAME>__API_KEY to use (secrets live in .env only).
    credentials: Credentials | None = None
    max_tokens: int = Field(default=16_000, ge=1_000)
    timeout_s: float = Field(default=300.0, gt=0)
    claude_bin: str = "claude"
    # Extra request fields for OpenAI-compatible providers, e.g. thinking mode.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # USD per million tokens: input, output, cache_read — for the session cost estimate.
    price_per_mtok: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _openai_needs_endpoint(self) -> Self:
        if self.backend == "openai" and (not self.base_url or not self.credentials):
            raise ValueError("an openai-compatible profile needs base_url and credentials")
        return self


def _default_models() -> dict[str, ModelProfile]:
    return {
        "claude-cli": ModelProfile(backend="claude-code", model="claude-opus-5-5", effort="medium")
    }


class AgentSettings(BaseModel):
    # Which [models.<name>] profile runs the trader (`enduro agent --model NAME` overrides).
    model_profile: str = "claude-cli"
    prompt_path: Path = Path("prompts/trader.md")
    journal_dir: Path = Path("state/journal")
    search_interval_s: int = Field(default=180, ge=15)
    focus_interval_s: int = Field(default=60, ge=15)
    max_llm_calls_per_tick: int = Field(default=8, ge=1)
    max_tool_calls_per_tick: int = Field(default=16, ge=2)
    wake_move_bps: float = Field(default=30.0, gt=0)
    wake_move_atr: float = Field(default=0.5, gt=0)
    wake_flat_multiplier: float = Field(default=2.0, ge=1)
    min_wake_gap_flat_s: int = Field(default=60, ge=0)
    min_wake_gap_position_s: int = Field(default=15, ge=0)
    min_check_flat_s: int = Field(default=120, ge=15)
    # Open interest and funding in the focus view and the prompt (prompts/positioning.md).
    show_positioning: bool = True


class ApiKey(BaseModel):
    """ENDURO_<PROVIDER>__API_KEY in .env; for Anthropic, the SDK's own resolution is the
    fallback."""

    api_key: SecretStr | None = None


class ApiCredentials(BaseModel):
    """Set via env only: ENDURO_BYBIT__API_KEY / ENDURO_BYBIT__API_SECRET."""

    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    def require(self) -> tuple[str, str]:
        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials are not set (see .env.example)")
        return self.api_key.get_secret_value(), self.api_secret.get_secret_value()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ENDURO_",
        env_nested_delimiter="__",
        env_file=".env",
        toml_file="config.toml",
        extra="ignore",
    )

    market: MarketConfig = Field(default_factory=MarketConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    bybit: ApiCredentials = Field(default_factory=ApiCredentials)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    models: dict[str, ModelProfile] = Field(default_factory=_default_models)
    anthropic: ApiKey = Field(default_factory=ApiKey)
    deepseek: ApiKey = Field(default_factory=ApiKey)  # ENDURO_DEEPSEEK__API_KEY
    glm: ApiKey = Field(default_factory=ApiKey)  # ENDURO_GLM__API_KEY

    @model_validator(mode="after")
    def _model_profile_exists(self) -> Self:
        if self.agent.model_profile not in self.models:
            raise ValueError(
                f"agent.model_profile={self.agent.model_profile!r} is not in "
                f"[models]: {sorted(self.models)}"
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
        )
