from __future__ import annotations

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class SessionConfig(BaseModel):
    london: tuple[int, int] = (8, 17)
    new_york: tuple[int, int] = (13, 22)
    asia: tuple[int, int] = (0, 9)
    enabled: list[str] = ["london", "new_york"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bingx_api_key: str = ""
    bingx_api_secret: str = ""

    top_pairs_count: int = 30
    scan_interval_seconds: int = 300
    min_daily_volume_usd: float = 10_000_000
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

    active_sessions: list[str] = ["london", "new_york"]

    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]


def is_active_session(config: SessionConfig) -> bool:
    from datetime import datetime, timezone

    hour = datetime.now(timezone.utc).hour
    for name in config.enabled:
        session = getattr(config, name, None)
        if session and session[0] <= hour < session[1]:
            return True
    return False
