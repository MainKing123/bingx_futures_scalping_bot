from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.schemas.common import TrendDirection


class SwingPoint(BaseModel):
    timestamp: datetime
    price: float
    type: Literal["HH", "HL", "LH", "LL"]
    index: int


class Trend(BaseModel):
    direction: TrendDirection
    last_swing_high: float | None = None
    last_swing_low: float | None = None


class StructureBreak(BaseModel):
    timestamp: datetime
    price: float
    type: Literal["BOS", "CHOCH"]
    direction: Literal["BULLISH", "BEARISH"]


class OrderBlock(BaseModel):
    timestamp: datetime
    zone_high: float
    zone_low: float
    type: Literal["BULLISH", "BEARISH"]
    mitigated: bool = False
    strength: float


class FairValueGap(BaseModel):
    timestamp: datetime
    zone_high: float
    zone_low: float
    type: Literal["BULLISH", "BEARISH"]
    fill_percent: float = 0.0


class LiquidityLevel(BaseModel):
    price: float
    type: Literal["EQUAL_HIGHS", "EQUAL_LOWS"]
    points: list[int]


class LiquiditySweep(BaseModel):
    timestamp: datetime
    level: float
    direction: Literal["BULLISH", "BEARISH"]


class PremiumDiscount(BaseModel):
    swing_high: float
    swing_low: float
    equilibrium: float
    premium_top: float
    premium_bottom: float
    discount_top: float
    discount_bottom: float
