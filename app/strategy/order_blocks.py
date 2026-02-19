from __future__ import annotations

import pandas as pd

from app.schemas.market import OrderBlock, StructureBreak


def find_order_blocks(df: pd.DataFrame, structure_breaks: list[StructureBreak], max_age: int = 50) -> list[OrderBlock]:
    result: list[OrderBlock] = []
    for br in structure_breaks:
        i = df.index.get_indexer([pd.Timestamp(br.timestamp)], method="nearest")[0]
        start = max(0, i - max_age)
        subset = df.iloc[start:i]
        if subset.empty:
            continue
        if br.direction == "BULLISH":
            cands = subset[subset["close"] < subset["open"]]
            ob_type = "BULLISH"
        else:
            cands = subset[subset["close"] > subset["open"]]
            ob_type = "BEARISH"
        if cands.empty:
            continue
        row = cands.iloc[-1]
        size = abs(float(row["open"] - row["close"]))
        impulse = abs(float(df.iloc[i]["close"] - row["close"]))
        if size <= 0 or impulse < 2 * size:
            continue
        result.append(OrderBlock(timestamp=cands.index[-1].to_pydatetime(), zone_high=float(max(row["open"], row["close"])), zone_low=float(min(row["open"], row["close"])), type=ob_type, strength=impulse / size))
    return result


def check_ob_mitigation(ob: OrderBlock, df: pd.DataFrame) -> bool:
    after = df[df.index > pd.Timestamp(ob.timestamp)]
    if after.empty:
        return False
    if ob.type == "BULLISH":
        return bool((after["close"] < ob.zone_low).any())
    return bool((after["close"] > ob.zone_high).any())
