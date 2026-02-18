from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Bias(str, Enum):
    bullish = "bullish"
    bearish = "bearish"
    neutral = "neutral"


class PositionSide(str, Enum):
    long = "long"
    short = "short"


class RiskConfig(BaseModel):
    equity_usdt: float = Field(default=1000, gt=0)
    risk_per_trade_pct: float = Field(default=0.5, gt=0, le=5)
    max_leverage: int = Field(default=10, ge=1, le=125)
    stop_buffer_pct: float = Field(default=0.1, ge=0, le=3)
    rr_target: float = Field(default=2.0, gt=0.5, le=10)
    daily_loss_limit_pct: float = Field(default=3.0, gt=0.1, le=20)
    max_consecutive_losses: int = Field(default=3, ge=1, le=20)
    cooldown_minutes: int = Field(default=10, ge=0, le=240)


class StrategyConfig(BaseModel):
    premium_zone: float = Field(default=0.65, gt=0.5, lt=1)
    discount_zone: float = Field(default=0.35, gt=0, lt=0.5)
    min_displacement_pct: float = Field(default=0.12, gt=0.01, le=2)
    min_rr: float = Field(default=1.5, ge=1.0, le=10)


class MarketTick(BaseModel):
    symbol: str = "BTC-USDT"
    price: float = Field(gt=0)
    high_1m: float = Field(gt=0)
    low_1m: float = Field(gt=0)
    high_30m: float = Field(gt=0)
    low_30m: float = Field(gt=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StrategySignal(BaseModel):
    has_signal: bool
    reason: str
    bias: Bias = Bias.neutral
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    side: Optional[PositionSide] = None


class Position(BaseModel):
    symbol: str
    side: PositionSide
    entry_price: float
    stop_price: float
    initial_stop_price: float
    take_profit: float
    tp1_price: float
    size_usdt: float
    open_size_usdt: float
    leverage: int
    margin_used: float
    moved_to_breakeven: bool = False
    trailing_active: bool = False
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BotStats(BaseModel):
    closed_pnl_usdt: float = 0.0
    consecutive_losses: int = 0
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0
    daily_loss_used_pct: float = 0.0
    cooldown_until: Optional[datetime] = None


class BotState(BaseModel):
    risk_config: RiskConfig
    strategy_config: StrategyConfig
    latest_signal: StrategySignal
    active_position: Optional[Position] = None
    stats: BotStats = Field(default_factory=BotStats)
