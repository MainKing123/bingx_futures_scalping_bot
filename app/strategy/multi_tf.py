from __future__ import annotations

from datetime import datetime, timezone

from app.config import SessionConfig, Settings, is_active_session
from app.exchange.client import BingXClient
from app.risk.risk_manager import RiskManager
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
    fvgs = [f for f in find_fvg(df)]
    zones = get_premium_discount_zones(swings)

    poi = []
    if trend.direction == TrendDirection.BULLISH:
        poi.extend([o for o in obs if o.type == "BULLISH" and is_in_discount(o.zone_low, zones)])
        poi.extend([f for f in fvgs if f.type == "BULLISH" and is_in_discount(f.zone_low, zones)])
    elif trend.direction == TrendDirection.BEARISH:
        poi.extend([o for o in obs if o.type == "BEARISH" and is_in_premium(o.zone_high, zones)])
        poi.extend([f for f in fvgs if f.type == "BEARISH" and is_in_premium(f.zone_high, zones)])

    return HTFAnalysis(trend=trend, bias=trend.direction, poi_zones=poi[:3], structure=structure, swings=swings, obs=obs, fvgs=fvgs)


async def find_ltf_entry(client: BingXClient, symbol: str, htf: HTFAnalysis, settings: Settings, risk_manager: RiskManager) -> TradeSetup | None:
    if htf.bias == TrendDirection.RANGING or not htf.poi_zones:
        return None
    if not risk_manager.can_open_setup():
        return None

    session_cfg = SessionConfig(enabled=settings.active_sessions)
    if not is_active_session(session_cfg):
        return None

    df = await client.get_klines(symbol, settings.ltf_timeframe, 200)
    swings = find_swing_points(df, settings.swing_lookback)
    trend = determine_trend(swings)
    choch = detect_choch(df, swings, trend)
    if not choch:
        return None

    choch_break = choch[-1]
    local_obs = find_order_blocks(df, [choch_break], max_age=settings.ob_max_age_candles)
    if not local_obs:
        return None
    ob = local_obs[-1]

    entry = (ob.zone_high + ob.zone_low) / 2
    if htf.bias == TrendDirection.BULLISH and ob.type != "BULLISH":
        return None
    if htf.bias == TrendDirection.BEARISH and ob.type != "BEARISH":
        return None

    choch_idx = df.index.get_indexer([choch_break.timestamp], method="nearest")[0]

    if htf.bias == TrendDirection.BULLISH:
        sl_candidates = [s.price for s in swings if s.index <= choch_idx and s.type in {"HL", "LL"}]
        sl = min(sl_candidates) if sl_candidates else ob.zone_low
        tps = [entry + (entry - sl) * x for x in (2, 3, 4)]
        direction = "LONG"
    else:
        sl_candidates = [s.price for s in swings if s.index <= choch_idx and s.type in {"HH", "LH"}]
        sl = max(sl_candidates) if sl_candidates else ob.zone_high
        tps = [entry - (sl - entry) * x for x in (2, 3, 4)]
        direction = "SHORT"

    risk = abs(entry - sl)
    if risk <= 0:
        return None
    rr = abs((tps[1] - entry) / risk)
    if rr < settings.min_risk_reward:
        return None

    confluences = ["HTF POI", "1m CHOCH", "1m OB"]
    if direction == "LONG" and any(x.type == "BULLISH" for x in htf.fvgs):
        confluences.append("Bullish FVG")
    if direction == "SHORT" and any(x.type == "BEARISH" for x in htf.fvgs):
        confluences.append("Bearish FVG")
    confluences.append("Active trading session")

    if len(confluences) >= 4:
        confidence = "HIGH"
    elif len(confluences) >= 3:
        confidence = "MEDIUM"
    elif len(confluences) >= settings.min_confluences:
        confidence = "LOW"
    else:
        return None

    position_size = risk_manager.calculate_position_size(entry, sl, settings.risk_per_trade_percent, settings.account_balance_usdt)

    return TradeSetup(
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        direction=direction,
        setup_type="CHOCH_OB",
        htf_bias=htf.bias.value,
        entry=float(entry),
        stop_loss=float(sl),
        take_profits=[float(x) for x in tps],
        risk_reward=float(rr),
        confidence=confidence,
        confluences=confluences,
        position_size_usdt=position_size,
    )
