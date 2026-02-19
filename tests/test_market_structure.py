from datetime import datetime, timezone

import pandas as pd

from app.schemas.common import TrendDirection
from app.schemas.market import SwingPoint, Trend
from app.strategy.market_structure import detect_bos, detect_choch, find_swing_points


def _df(values: list[tuple[float, float, float, float]]):
    idx = pd.date_range(datetime(2025, 1, 1, tzinfo=timezone.utc), periods=len(values), freq="min")
    return pd.DataFrame(values, columns=["open", "high", "low", "close"], index=idx)


def test_find_swing_points_detects_high_and_low():
    df = _df([(1, 2, 1, 1.5), (1.5, 3, 1.4, 2.8), (2.8, 2.9, 1.2, 1.3), (1.3, 3.2, 1.2, 3.0), (3.0, 3.1, 1.0, 1.1)])
    swings = find_swing_points(df, lookback=1)
    assert len(swings) >= 2


def test_detect_bos_returns_first_crossing_only():
    df = _df([(10, 11, 9, 10), (10, 11, 9, 10.5), (10.5, 12, 10, 11.2), (11.2, 13, 11, 12.5), (12.5, 14, 12, 13)])
    trend = Trend(direction=TrendDirection.BULLISH, last_swing_high=11.0, last_swing_low=9.5)
    out = detect_bos(df, [], trend)
    assert len(out) == 1
    assert out[0].price == 11.2


def test_detect_choch_returns_first_crossing_only():
    df = _df([(10, 11, 9, 10.5), (10.5, 11, 10, 10.8), (10.8, 11, 10, 9.8), (9.8, 10, 9, 9.4)])
    swings = [SwingPoint(timestamp=df.index[0].to_pydatetime(), price=10.0, type="HL", index=0)]
    trend = Trend(direction=TrendDirection.BULLISH, last_swing_high=11.0, last_swing_low=10.0)
    out = detect_choch(df, swings, trend)
    assert len(out) == 1
    assert out[0].price == 9.8
