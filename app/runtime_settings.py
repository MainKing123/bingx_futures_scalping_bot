"""Current runtime policy; historical Settings and signed studies stay frozen."""
from __future__ import annotations

from typing import Literal
from pydantic import Field, field_validator, model_validator
from app.config import Settings


class RuntimeSettings(Settings):
    volium_strategy_profile: Literal["v1_guarded", "v5_strict", "v5_equal"] = "v5_equal"
    trading_symbols: list[str] = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT", "DOGE_USDT"]
    universe_core_symbols: list[str] = ["BTC_USDT", "ETH_USDT"]
    default_leverage: int = Field(default=50, ge=10, le=50)
    leverage_asset_caps: dict[str, int] = {
        "BTC_USDT": 50, "ETH_USDT": 50, "SOL_USDT": 25,
        "XRP_USDT": 25, "DOGE_USDT": 10,
    }
    execution_cost_guard_enabled: bool = True
    min_net_risk_reward: float = Field(default=1.25, ge=1, le=5)
    max_cost_to_price_risk: float = Field(default=1 / 3, gt=0, le=1)
    max_stop_distance_percent: float = Field(default=1.5, gt=0, le=10)
    liquidation_distance_safety_fraction: float = Field(default=.8, gt=0, lt=1)
    liquidation_gap_buffer_bps: float = Field(default=10, ge=0, le=500)
    fair_price_basis_buffer_bps: float = Field(default=20, ge=0, le=500)
    adverse_funding_reserve_bps: float = Field(default=10, ge=0, le=500)

    @field_validator("universe_core_symbols")
    @classmethod
    def validate_core_symbols(cls, values):
        if not values:
            return []
        if len(values) > 5 or cls.validate_symbols(values) != values:
            raise ValueError("Use up to five unique canonical SYMBOL_USDT core pairs")
        return values

    @field_validator("leverage_asset_caps")
    @classmethod
    def validate_leverage_caps(cls, values):
        if cls.validate_symbols(list(values)) != list(values):
            raise ValueError("Use canonical uppercase SYMBOL_USDT leverage keys")
        if any(isinstance(value, bool) or not isinstance(value, int) or not 10 <= value <= 50
               for value in values.values()):
            raise ValueError("Asset leverage caps must be integers from 10 to 50")
        return values

    @model_validator(mode="after")
    def validate_research_profile(self):
        if self.volium_strategy_profile.startswith("v5"):
            if self.auto_execution:
                raise ValueError("Experimental V5 supports paper trading only")
            if self.volium_mode not in {"intraday", "scalp"}:
                raise ValueError("V5 source profile supports intraday/scalp only")
        return self
