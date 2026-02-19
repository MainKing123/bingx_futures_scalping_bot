from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LiquiditySnapshot(BaseModel):
    bsl_levels: list[float] = Field(default_factory=list)
    ssl_levels: list[float] = Field(default_factory=list)
    nearest_bsl: float | None = None
    nearest_ssl: float | None = None
    summary: str


class CRTSnapshot(BaseModel):
    detected: bool
    direction: Literal["LONG", "SHORT"] | None = None
    range_high: float | None = None
    range_low: float | None = None
    sweep_level: float | None = None
    close_inside: bool = False
    rejection_ratio: float | None = None
    summary: str


class ICTSnapshot(BaseModel):
    htf_4h_bias: Literal["BULLISH", "BEARISH", "RANGING"]
    htf_1d_bias: Literal["BULLISH", "BEARISH", "RANGING"]
    htf_aligned: bool
    mss_confirmed: bool
    ob_confirmed: bool
    fvg_confirmed: bool
    confirmations: list[str] = Field(default_factory=list)
    summary: str


class SessionSnapshot(BaseModel):
    killzone_enabled: bool
    killzone_active: bool
    active_killzone: Literal["london", "new_york"] | None = None
    summary: str


class CRTICTAnalysisResponse(BaseModel):
    symbol: str
    timeframe: str
    timestamp: datetime
    liquidity: LiquiditySnapshot
    crt: CRTSnapshot
    ict: ICTSnapshot
    session: SessionSnapshot
    signal: Literal["LONG", "SHORT", "NO_SIGNAL"]
    entry: float | None = None
    stop: float | None = None
    targets: list[float] = Field(default_factory=list)
    rr: float | None = None
