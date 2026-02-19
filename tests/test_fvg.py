from datetime import datetime, timezone

import pandas as pd

from app.strategy.fvg import find_fvg


def test_find_fvg_detects_bullish_gap():
    idx = pd.date_range(datetime(2025, 1, 1, tzinfo=timezone.utc), periods=5, freq="min")
    df = pd.DataFrame(
        {
            "open": [100, 101, 106, 107, 108],
            "high": [101, 102, 107, 108, 109],
            "low": [99, 100, 105, 106, 107],
            "close": [100.5, 101.5, 106.5, 107.5, 108.5],
        },
        index=idx,
    )
    fvgs = find_fvg(df, fvg_min_size_percent=0.1)
    assert len(fvgs) >= 1
    assert any(f.type == "BULLISH" for f in fvgs)
