from __future__ import annotations

from datetime import datetime, timezone

from app.exchange.client import BingXClient
from app.schemas.common import TrendDirection
from app.schemas.setup import HTFAnalysis, TradeSetup
from app.strategy.fvg import find_fvg
from app.strategy.market_structure import detect_bos, detect_choch, determine_trend, find_swing_points
from app.strategy.order_blocks import check_ob_mitigation, find_order_blocks
from app.strategy.premium_discount import get_premium_discount_zones, is_in_discount, is_in_premium


async def analyze_htf(client: BingXClient, symbol: str) -> HTFAnalysis:
    df = await client.get_klines(symbol, "30m", 200)
    swings = find_swing_points(df)
    trend = determine_trend(swings)
    structure = detect_bos(df, swings, trend) + detect_choch(df, swings, trend)
    obs = [x for x in find_order_blocks(df, structure) if not check_ob_mitigation(x, df)]
    fvgs = [f for f in find_fvg(df) if f.fill_percent < 0.5]
    zones = get_premium_discount_zones(swings)

    poi = []
    if trend.direction == TrendDirection.BULLISH:
        poi.extend([o for o in obs if o.type == "BULLISH" and is_in_discount(o.zone_low, zones)])
        poi.extend([f for f in fvgs if f.type == "BULLISH" and is_in_discount(f.zone_low, zones)])
    elif trend.direction == TrendDirection.BEARISH:
        poi.extend([o for o in obs if o.type == "BEARISH" and is_in_premium(o.zone_high, zones)])
        poi.extend([f for f in fvgs if f.type == "BEARISH" and is_in_premium(f.zone_high, zones)])

    return HTFAnalysis(trend=trend, bias=trend.direction, poi_zones=poi[:3], structure=structure, swings=swings, obs=obs, fvgs=fvgs)


async def find_ltf_entry(client: BingXClient, symbol: str, htf: HTFAnalysis) -> TradeSetup | None:
    if htf.bias == TrendDirection.RANGING or not htf.poi_zones:
        return None
    df = await client.get_klines(symbol, "1m", 200)
    swings = find_swing_points(df)
    trend = determine_trend(swings)
    choch = detect_choch(df, swings, trend)
    if not choch:
        return None

    last = df.iloc[-1]
    entry = float(last["close"])
    if htf.bias == TrendDirection.BULLISH:
        sl = min([s.price for s in swings[-10:] if s.type in {"HL", "LL"}] or [entry * 0.995])
        tps = [entry + (entry - sl) * x for x in (2, 3, 4)]
        direction = "LONG"
    else:
        sl = max([s.price for s in swings[-10:] if s.type in {"HH", "LH"}] or [entry * 1.005])
        tps = [entry - (sl - entry) * x for x in (2, 3, 4)]
        direction = "SHORT"
    rr = abs((tps[1] - entry) / (entry - sl)) if direction == "LONG" else abs((entry - tps[1]) / (sl - entry))
    if rr < 3:
        return None
    confluences = ["HTF POI", "1m CHOCH", "Session filter"]
    confidence = "MEDIUM"
    return TradeSetup(timestamp=datetime.now(timezone.utc), symbol=symbol, direction=direction, setup_type="CHOCH_OB", htf_bias=htf.bias.value, entry=entry, stop_loss=float(sl), take_profits=[float(x) for x in tps], risk_reward=float(rr), confidence=confidence, confluences=confluences)
