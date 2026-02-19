from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel


class CRTCandle(BaseModel):
    timestamp: datetime
    direction: Literal["LONG", "SHORT"]
    range_high: float
    range_low: float
    sweep_level: float
    close_inside: bool
    rejection_ratio: float


def _wick_body_ratio(row: pd.Series, direction: Literal["LONG", "SHORT"]) -> float:
    open_price = float(row["open"])
    close_price = float(row["close"])
    high_price = float(row["high"])
    low_price = float(row["low"])
    body = max(abs(close_price - open_price), 1e-9)
    if direction == "SHORT":
        wick = max(0.0, high_price - max(open_price, close_price))
    else:
        wick = max(0.0, min(open_price, close_price) - low_price)
    return wick / body


def detect_crt_candle(
    df: pd.DataFrame,
    lookback: int = 20,
    min_sweep_pct: float = 0.03,
    min_wick_body_ratio: float = 1.2,
) -> CRTCandle | None:
    if df.empty or len(df) < lookback + 2:
        return None

    current = df.iloc[-1]
    history = df.iloc[-(lookback + 1) : -1]
    if history.empty:
        return None

    range_high = float(history["high"].max())
    range_low = float(history["low"].min())
    high = float(current["high"])
    low = float(current["low"])
    open_price = float(current["open"])
    close_price = float(current["close"])
    close_inside = range_low <= close_price <= range_high
    if not close_inside:
        return None

    min_sweep_frac = max(0.0, float(min_sweep_pct) / 100.0)
    high_sweep = high >= range_high * (1 + min_sweep_frac)
    low_sweep = low <= range_low * (1 - min_sweep_frac)

    direction: Literal["LONG", "SHORT"] | None = None
    sweep_level = 0.0
    if high_sweep and not low_sweep:
        direction = "SHORT"
        sweep_level = high
    elif low_sweep and not high_sweep:
        direction = "LONG"
        sweep_level = low
    elif high_sweep and low_sweep:
        high_exc = (high - range_high) / max(range_high, 1e-9)
        low_exc = (range_low - low) / max(abs(range_low), 1e-9)
        if high_exc >= low_exc:
            direction = "SHORT"
            sweep_level = high
        else:
            direction = "LONG"
            sweep_level = low
    else:
        return None

    # Require candle body to cross back into the prior range (rejection body).
    if direction == "SHORT" and open_price <= range_high:
        return None
    if direction == "LONG" and open_price >= range_low:
        return None

    rejection_ratio = _wick_body_ratio(current, direction)
    if rejection_ratio < float(min_wick_body_ratio):
        return None

    return CRTCandle(
        timestamp=df.index[-1].to_pydatetime(),
        direction=direction,
        range_high=range_high,
        range_low=range_low,
        sweep_level=sweep_level,
        close_inside=close_inside,
        rejection_ratio=float(rejection_ratio),
    )
