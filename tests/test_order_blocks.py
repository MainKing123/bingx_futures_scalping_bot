from datetime import datetime, timezone

import pandas as pd

from app.schemas.market import StructureBreak
from app.strategy.order_blocks import check_ob_mitigation, find_order_blocks


def test_find_order_blocks_and_mitigation_rule():
    idx = pd.date_range(datetime(2025, 1, 1, tzinfo=timezone.utc), periods=6, freq="min")
    df = pd.DataFrame(
        {
            "open": [100, 99, 98, 102, 105, 106],
            "high": [101, 100, 103, 106, 107, 108],
            "low": [99, 97, 97, 101, 104, 96],
            "close": [99, 98, 102, 105, 106, 97],
        },
        index=idx,
    )
    br = [StructureBreak(timestamp=idx[4].to_pydatetime(), price=106, type="BOS", direction="BULLISH")]
    obs = find_order_blocks(df, br, max_age=5)
    assert obs
    ob = obs[0]
    assert check_ob_mitigation(ob, df) is True
