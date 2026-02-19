from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    min_risk_reward: float
    swing_lookback: int
    ob_max_age_candles: int
    min_confluences: int
    max_poi_distance_pct: float


def build_profile(name: str, settings: Settings) -> StrategyProfile:
    profiles: dict[str, StrategyProfile] = {
        "default": StrategyProfile(
            name="default",
            min_risk_reward=settings.min_risk_reward,
            swing_lookback=settings.swing_lookback,
            ob_max_age_candles=settings.ob_max_age_candles,
            min_confluences=settings.min_confluences,
            max_poi_distance_pct=settings.max_poi_distance_pct,
        ),
        "conservative": StrategyProfile(
            name="conservative",
            min_risk_reward=max(3.5, settings.min_risk_reward),
            swing_lookback=max(4, settings.swing_lookback),
            ob_max_age_candles=min(35, settings.ob_max_age_candles),
            min_confluences=max(3, settings.min_confluences),
            max_poi_distance_pct=min(0.35, settings.max_poi_distance_pct),
        ),
        "aggressive": StrategyProfile(
            name="aggressive",
            min_risk_reward=min(2.0, settings.min_risk_reward),
            swing_lookback=max(2, settings.swing_lookback - 1),
            ob_max_age_candles=max(60, settings.ob_max_age_candles),
            min_confluences=max(2, settings.min_confluences - 1),
            max_poi_distance_pct=max(0.75, settings.max_poi_distance_pct),
        ),
        "high_rr": StrategyProfile(
            name="high_rr",
            min_risk_reward=max(4.0, settings.min_risk_reward),
            swing_lookback=max(4, settings.swing_lookback),
            ob_max_age_candles=min(45, settings.ob_max_age_candles),
            min_confluences=max(3, settings.min_confluences),
            max_poi_distance_pct=min(0.4, settings.max_poi_distance_pct),
        ),
    }
    return profiles.get(name, profiles["default"])
