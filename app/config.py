from __future__ import annotations

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class SessionConfig(BaseModel):
    london: tuple[int, int] = (8, 17)
    new_york: tuple[int, int] = (13, 22)
    asia: tuple[int, int] = (0, 9)
    enabled: list[str] = ["all"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bingx_api_key: str = ""
    bingx_api_secret: str = ""

    top_pairs_count: int = 20
    scan_interval_seconds: int = 300
    min_daily_volume_usd: float = 10_000_000
    volatility_pool_size: int = 60
    volatility_lookback_candles: int = 96
    volatility_interval: str = "5m"
    max_poi_distance_pct: float = 0.5
    watchlist_max_age_hours: int = 4

    htf_timeframe: str = "30m"
    ltf_timeframe: str = "1m"
    min_risk_reward: float = 3.0
    swing_lookback: int = 3
    ob_max_age_candles: int = 50
    fvg_min_size_percent: float = 0.1
    min_confluences: int = 2

    risk_per_trade_percent: float = 1.0
    max_open_setups: int = 5
    daily_loss_limit_percent: float = 3.0
    account_balance_usdt: float = 1000.0

    auto_execution: bool = False
    default_leverage: int = 10

    backtest_fee_bps: float = 5.0
    backtest_slippage_bps: float = 2.0
    backtest_cooldown_candles: int = 3
    backtest_default_lookback_days: int = 14
    backtest_max_lookback_days: int = 30

    active_sessions: list[str] = ["all"]

    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]


def is_active_session(config: SessionConfig) -> bool:
    from datetime import datetime, timezone

    enabled = {name.strip().lower() for name in config.enabled if isinstance(name, str)}
    if not enabled or "all" in enabled:
        return True

    hour = datetime.now(timezone.utc).hour
    for name in enabled:
        session = getattr(config, name, None)
        if session and session[0] <= hour < session[1]:
            return True
    return False
