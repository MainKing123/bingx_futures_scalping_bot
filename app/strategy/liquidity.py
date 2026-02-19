from __future__ import annotations

import pandas as pd

from app.schemas.market import LiquidityLevel, LiquiditySweep, SwingPoint


def find_equal_levels(swings: list[SwingPoint], tolerance: float = 0.0005) -> list[LiquidityLevel]:
    highs = [s for s in swings if s.type in {"HH", "LH"}]
    lows = [s for s in swings if s.type in {"HL", "LL"}]
    levels: list[LiquidityLevel] = []
    for group, typ in ((highs, "EQUAL_HIGHS"), (lows, "EQUAL_LOWS")):
        for i, a in enumerate(group):
            cluster = [a.index]
            for b in group[i + 1:]:
                if abs(a.price - b.price) / a.price < tolerance:
                    cluster.append(b.index)
            if len(cluster) >= 2:
                levels.append(LiquidityLevel(price=a.price, type=typ, points=cluster))
    return levels


def detect_liquidity_sweep(df: pd.DataFrame, levels: list[LiquidityLevel]) -> list[LiquiditySweep]:
    sweeps: list[LiquiditySweep] = []
    for lvl in levels:
        if lvl.type == "EQUAL_HIGHS":
            mask = (df["high"] > lvl.price) & (df["close"] < lvl.price)
            direction = "BEARISH"
        else:
            mask = (df["low"] < lvl.price) & (df["close"] > lvl.price)
            direction = "BULLISH"
        for ts in df[mask].index:
            sweeps.append(LiquiditySweep(timestamp=ts.to_pydatetime(), level=lvl.price, direction=direction))
    return sweeps
