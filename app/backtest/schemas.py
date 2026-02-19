from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BacktestRunRequest(BaseModel):
    mode: Literal["single", "batch"] = "single"
    strategy: Literal["legacy_choch_ob", "crt_ict"] = "crt_ict"
    symbol: str | None = None
    lookback_days: int | None = None
    ltf_timeframe: str | None = None
    htf_timeframe: str | None = None
    profile: Literal["default", "conservative", "aggressive", "high_rr"] = "default"

    @model_validator(mode="after")
    def validate_payload(self) -> "BacktestRunRequest":
        if self.mode == "single" and not self.symbol:
            raise ValueError("symbol is required for single mode")
        return self


class BacktestTradeLogItem(BaseModel):
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    direction: Literal["LONG", "SHORT"]
    status: str
    entry_price: float
    exit_price: float
    pnl_percent: float


class BacktestSymbolResult(BaseModel):
    symbol: str
    trades_count: int
    wins: int
    losses: int
    win_rate: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    total_pnl_percent: float
    trade_log: list[BacktestTradeLogItem] = Field(default_factory=list)


class BacktestSummary(BaseModel):
    job_id: str
    mode: Literal["single", "batch"]
    strategy: str
    profile: str
    lookback_days: int
    ltf_timeframe: str
    htf_timeframe: str
    htf_context: str
    universe: list[str]
    started_at: datetime
    finished_at: datetime
    trades_count: int
    wins: int
    losses: int
    win_rate: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    total_pnl_percent: float
    symbol_results: list[BacktestSymbolResult] = Field(default_factory=list)


class BacktestRunStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    mode: Literal["single", "batch"]
    strategy: str
    profile: str
    progress: float
    created_at: datetime
    updated_at: datetime
    error: str | None = None
