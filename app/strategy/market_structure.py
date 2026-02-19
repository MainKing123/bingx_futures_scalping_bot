from __future__ import annotations

import pandas as pd

from app.schemas.common import TrendDirection
from app.schemas.market import StructureBreak, SwingPoint, Trend


def find_swing_points(df: pd.DataFrame, lookback: int = 3) -> list[SwingPoint]:
    highs = df["high"]
    lows = df["low"]
    roll_max = highs.rolling(2 * lookback + 1, center=True).max()
    roll_min = lows.rolling(2 * lookback + 1, center=True).min()
    swing_high_idx = highs.eq(roll_max)
    swing_low_idx = lows.eq(roll_min)
    swings: list[SwingPoint] = []
    last_high = None
    last_low = None
    for i, ts in enumerate(df.index):
        if bool(swing_high_idx.iloc[i]):
            price = float(highs.iloc[i])
            s_type = "HH" if last_high is None or price >= last_high else "LH"
            last_high = price
            swings.append(SwingPoint(timestamp=ts.to_pydatetime(), price=price, type=s_type, index=i))
        if bool(swing_low_idx.iloc[i]):
            price = float(lows.iloc[i])
            s_type = "LL" if last_low is None or price <= last_low else "HL"
            last_low = price
            swings.append(SwingPoint(timestamp=ts.to_pydatetime(), price=price, type=s_type, index=i))
    swings.sort(key=lambda x: x.index)
    return swings


def determine_trend(swings: list[SwingPoint]) -> Trend:
    if len(swings) < 4:
        return Trend(direction=TrendDirection.RANGING)
    recent = swings[-6:]
    hh = sum(s.type == "HH" for s in recent)
    hl = sum(s.type == "HL" for s in recent)
    lh = sum(s.type == "LH" for s in recent)
    ll = sum(s.type == "LL" for s in recent)
    if hh + hl > lh + ll:
        return Trend(direction=TrendDirection.BULLISH, last_swing_high=max((s.price for s in recent if s.type in {"HH", "LH"}), default=None), last_swing_low=max((s.price for s in recent if s.type in {"HL", "LL"}), default=None))
    if lh + ll > hh + hl:
        return Trend(direction=TrendDirection.BEARISH, last_swing_high=max((s.price for s in recent if s.type in {"HH", "LH"}), default=None), last_swing_low=min((s.price for s in recent if s.type in {"HL", "LL"}), default=None))
    return Trend(direction=TrendDirection.RANGING)


def _first_crossing(df: pd.DataFrame, level: float, direction: str) -> tuple[pd.Timestamp, float] | None:
    if direction == "above":
        hits = df[df["close"] > level]
    else:
        hits = df[df["close"] < level]
    if hits.empty:
        return None
    ts = hits.index[0]
    return ts, float(hits.iloc[0]["close"])


def detect_bos(df: pd.DataFrame, swings: list[SwingPoint], trend: Trend) -> list[StructureBreak]:
    if trend.direction == TrendDirection.BULLISH and trend.last_swing_high is not None:
        hit = _first_crossing(df, trend.last_swing_high, "above")
        if hit:
            return [StructureBreak(timestamp=hit[0].to_pydatetime(), price=hit[1], type="BOS", direction="BULLISH")]
    if trend.direction == TrendDirection.BEARISH and trend.last_swing_low is not None:
        hit = _first_crossing(df, trend.last_swing_low, "below")
        if hit:
            return [StructureBreak(timestamp=hit[0].to_pydatetime(), price=hit[1], type="BOS", direction="BEARISH")]
    return []


def detect_choch(df: pd.DataFrame, swings: list[SwingPoint], trend: Trend) -> list[StructureBreak]:
    if trend.direction == TrendDirection.BEARISH:
        lvl = max((s.price for s in swings if s.type == "LH"), default=None)
        if lvl is not None:
            hit = _first_crossing(df, lvl, "above")
            if hit:
                return [StructureBreak(timestamp=hit[0].to_pydatetime(), price=hit[1], type="CHOCH", direction="BULLISH")]
    if trend.direction == TrendDirection.BULLISH:
        lvl = min((s.price for s in swings if s.type == "HL"), default=None)
        if lvl is not None:
            hit = _first_crossing(df, lvl, "below")
            if hit:
                return [StructureBreak(timestamp=hit[0].to_pydatetime(), price=hit[1], type="CHOCH", direction="BEARISH")]
    return []
