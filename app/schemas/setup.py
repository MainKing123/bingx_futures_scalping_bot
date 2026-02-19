from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.schemas.common import Candle, TrendDirection
from app.schemas.market import FairValueGap, OrderBlock, StructureBreak, SwingPoint, Trend


class ChartMarking(BaseModel):
    type: Literal[
        "order_block",
        "fvg",
        "swing_high",
        "swing_low",
        "bos",
        "choch",
        "entry",
        "stop_loss",
        "take_profit",
        "liquidity",
    ]
    start_time: datetime
    end_time: datetime | None = None
    price_top: float
    price_bottom: float
    label: str
    color: str


class ChartData(BaseModel):
    htf_candles: list[Candle] = []
    ltf_candles: list[Candle] = []
    markings: list[ChartMarking] = []


class TradeSetup(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime
    symbol: str
    direction: Literal["LONG", "SHORT"]
    setup_type: Literal["BOS_OB", "CHOCH_OB", "FVG_ENTRY", "LIQUIDITY_SWEEP"]
    htf_bias: Literal["BULLISH", "BEARISH"]
    entry: float
    stop_loss: float
    take_profits: list[float]
    risk_reward: float
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    confluences: list[str]
    status: Literal["ACTIVE", "TP1_HIT", "TP2_HIT", "TP3_HIT", "SL_HIT", "EXPIRED", "CANCELLED"] = "ACTIVE"
    position_size_usdt: float | None = None
    chart_data: ChartData | None = None


class HTFAnalysis(BaseModel):
    trend: Trend
    bias: TrendDirection
    poi_zones: list[OrderBlock | FairValueGap]
    structure: list[StructureBreak]
    swings: list[SwingPoint]
    obs: list[OrderBlock]
    fvgs: list[FairValueGap]


class SymbolAnalysis(BaseModel):
    symbol: str
    trend: Trend
    swings: list[SwingPoint]
    structure: list[StructureBreak]
    order_blocks: list[OrderBlock]
    fvgs: list[FairValueGap]


class MarketOverview(BaseModel):
    symbol: str
    trend: TrendDirection
    bias: TrendDirection
    active_obs: list[OrderBlock]
    active_fvgs: list[FairValueGap]
