from __future__ import annotations

import pandas as pd

from app.schemas.market import FairValueGap


def find_fvg(df: pd.DataFrame, fvg_min_size_percent: float = 0.1) -> list[FairValueGap]:
    bull = df["low"].shift(-1) > df["high"].shift(1)
    bear = df["high"].shift(-1) < df["low"].shift(1)
    out: list[FairValueGap] = []
    for i in range(1, len(df) - 1):
        if bull.iloc[i]:
            zh, zl = float(df["low"].iloc[i + 1]), float(df["high"].iloc[i - 1])
            if ((zh - zl) / df["close"].iloc[i]) * 100 >= fvg_min_size_percent:
                out.append(FairValueGap(timestamp=df.index[i].to_pydatetime(), zone_high=zh, zone_low=zl, type="BULLISH"))
        if bear.iloc[i]:
            zh, zl = float(df["low"].iloc[i - 1]), float(df["high"].iloc[i + 1])
            if ((zh - zl) / df["close"].iloc[i]) * 100 >= fvg_min_size_percent:
                out.append(FairValueGap(timestamp=df.index[i].to_pydatetime(), zone_high=zh, zone_low=zl, type="BEARISH"))
    return out


def check_fvg_fill(fvg: FairValueGap, df: pd.DataFrame) -> float:
    after = df[df.index > pd.Timestamp(fvg.timestamp)]
    if after.empty:
        return 0.0
    if fvg.type == "BULLISH":
        min_low = float(after["low"].min())
        fill = max(0.0, min(1.0, (fvg.zone_high - min_low) / max(1e-9, (fvg.zone_high - fvg.zone_low))))
    else:
        max_high = float(after["high"].max())
        fill = max(0.0, min(1.0, (max_high - fvg.zone_low) / max(1e-9, (fvg.zone_high - fvg.zone_low))))
    return fill
