"""Application settings.

Priority (highest first): constructor args → env vars (ENDURO_*) → .env → config.toml.
Secrets must only come from env / .env, never from config.toml.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator
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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ENDURO_",
        env_nested_delimiter="__",
        env_file=".env",
        toml_file="config.toml",
        extra="ignore",
    )

    market: MarketConfig = Field(default_factory=MarketConfig)

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
