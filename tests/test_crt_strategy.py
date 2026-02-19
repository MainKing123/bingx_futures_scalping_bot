from datetime import datetime, timezone

import pandas as pd

from app.strategy.crt import detect_crt_candle


def _frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range(datetime(2025, 1, 1, tzinfo=timezone.utc), periods=len(rows), freq="min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_detect_crt_false_break_and_close_inside():
    rows = [(100.0, 100.6, 99.7, 100.1)] * 8
    rows.append((101.2, 103.0, 99.8, 100.0))
    df = _frame(rows)
    crt = detect_crt_candle(df, lookback=5, min_sweep_pct=0.03, min_wick_body_ratio=1.2)
    assert crt is not None
    assert crt.direction == "SHORT"
    assert crt.close_inside is True


def test_detect_crt_allows_open_inside_range():
    rows = [(100.0, 100.6, 99.7, 100.1)] * 8
    rows.append((100.4, 102.9, 99.8, 100.0))
    df = _frame(rows)
    crt = detect_crt_candle(df, lookback=5, min_sweep_pct=0.03, min_wick_body_ratio=1.2)
    assert crt is not None
    assert crt.direction == "SHORT"


def test_detect_crt_requires_close_inside_range():
    rows = [(100.0, 100.5, 99.8, 100.1)] * 8
    rows.append((101.2, 103.0, 100.2, 102.0))
    df = _frame(rows)
    crt = detect_crt_candle(df, lookback=5, min_sweep_pct=0.03, min_wick_body_ratio=1.2)
    assert crt is None


def test_detect_crt_rejects_weak_wick_body_ratio():
    rows = [(100.0, 100.5, 99.8, 100.1)] * 8
    rows.append((101.1, 101.7, 99.9, 100.4))
    df = _frame(rows)
    crt = detect_crt_candle(df, lookback=5, min_sweep_pct=0.03, min_wick_body_ratio=2.0)
    assert crt is None
