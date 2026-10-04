from __future__ import annotations
from datetime import datetime
from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, Field


class TradeSetup(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:24])
    timestamp: datetime
    symbol: str
    direction: Literal["LONG", "SHORT"]
    setup_type: Literal["VOLIUM_INTRADAY", "VOLIUM_SCALP", "VOLIUM_SWING"]
    htf_bias: Literal["BULLISH", "BEARISH"]
    entry: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profits: list[float] = Field(min_length=1)
    risk_reward: float
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    confluences: list[str] = []
    status: Literal["ACTIVE", "TP1_HIT", "SL_HIT", "EXPIRED", "CANCELLED", "CLOSED"] = "ACTIVE"
    position_size_usdt: float | None = None
