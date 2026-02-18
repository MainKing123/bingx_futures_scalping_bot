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
    take_profit: float
    size_usdt: float
    leverage: int
    margin_used: float
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BotState(BaseModel):
    risk_config: RiskConfig
    latest_signal: StrategySignal
    active_position: Optional[Position] = None
    closed_pnl_usdt: float = 0.0
