from __future__ import annotations

import re
from typing import Literal
from zoneinfo import ZoneInfo
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", validate_assignment=True)
    mexc_api_key: str = Field(default="", repr=False, exclude=True)
    mexc_api_secret: str = Field(default="", repr=False, exclude=True)
    mexc_base_url: str = "https://api.mexc.com"
    mexc_receive_window: int = Field(default=10, ge=1, le=60)
    trading_symbols: list[str] = ["ZEC_USDT", "SOL_USDT", "DOGE_USDT", "XRP_USDT", "ETH_USDT"]
    pair_selection: Literal["dynamic", "fixed"] = "dynamic"
    pair_count: int = Field(default=5, ge=5, le=5)
    large_cap_top_n: int = Field(default=20, ge=10, le=100)
    min_pair_volume_usd: float = Field(default=50_000_000, ge=1_000_000)
    universe_refresh_seconds: int = Field(default=3600, ge=300, le=21600)
    scan_interval_seconds: int = Field(default=30, ge=5, le=3600)
    volium_mode: Literal["intraday", "scalp", "swing"] = "intraday"
    volium_swing_context: Literal["1d", "1w"] = "1d"
    volium_swing_lookback: int = Field(default=2, ge=1, le=10)
    volium_context_lookback: int = Field(default=80, ge=30, le=500)
    volium_sweep_max_age_bars: int = Field(default=12, ge=1, le=100)
    volium_reaction_max_bars: int = Field(default=3, ge=1, le=20)
    volium_min_body_ratio: float = Field(default=0.6, gt=0, le=1)
    volium_stop_buffer_bps: float = Field(default=2.0, ge=0, le=100)
    volium_scalp_stop_buffer_bps: float = Field(default=10.0, ge=0, le=100)
    volium_sessions_utc3: list[tuple[str, str]] = [("10:00", "12:00"), ("16:30", "18:00")]
    volium_session_clock: Literal["fixed_utc3", "market_local"] = "fixed_utc3"
    volium_market_sessions: list[tuple[str, str, str]] = [
        ("Europe/London", "08:00", "10:00"), ("America/New_York", "09:30", "11:00")]
    volium_session_enabled: bool = True
    volium_require_daily_origin_sweep: bool = True
    volium_active_trend_min_atr: float = Field(default=1.0, ge=0, le=10)
    risk_per_trade_percent: float = Field(default=0.5, gt=0, le=5)
    max_open_setups: int = Field(default=3, ge=1, le=20)
    daily_loss_limit_percent: float = Field(default=2.0, gt=0, le=20)
    account_balance_usdt: float = Field(default=1000.0, gt=0)
    auto_execution: bool = False
    default_leverage: int = Field(default=3, ge=1, le=20)
    pending_order_max_age_minutes: int = Field(default=30, ge=1, le=1440)
    paper_fee_bps: float = Field(default=5.0, ge=0, le=100)
    paper_slippage_bps: float = Field(default=2.0, ge=0, le=100)
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("trading_symbols")
    @classmethod
    def validate_symbols(cls, values):
        symbols = list(dict.fromkeys(v.upper().replace("-", "_") for v in values))
        if not symbols or any(not re.fullmatch(r"[A-Z0-9]+_USDT", v) for v in symbols):
            raise ValueError("Use USDT futures symbols, e.g. BTC_USDT")
        return symbols

    @field_validator("volium_sessions_utc3")
    @classmethod
    def validate_sessions(cls, values):
        for start, end in values:
            if any(not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", t) for t in (start, end)) or start == end:
                raise ValueError("Session windows require different HH:MM times")
        return values

    @field_validator("volium_market_sessions")
    @classmethod
    def validate_market_sessions(cls, values):
        for name, start, end in values:
            ZoneInfo(name)
            cls.validate_sessions([(start, end)])
        return values
