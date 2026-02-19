from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from app.config import Settings
from app.risk.risk_manager import RiskManager
from app.schemas.analysis import CRTICTAnalysisResponse, CRTSnapshot, ICTSnapshot, LiquiditySnapshot, SessionSnapshot
from app.schemas.common import TrendDirection
from app.schemas.setup import TradeSetup
from app.strategy.crt import detect_crt_candle
from app.strategy.fvg import find_fvg
from app.strategy.liquidity import find_equal_levels
from app.strategy.market_structure import detect_bos, detect_choch, determine_trend, find_swing_points
from app.strategy.multi_tf import analyze_htf_from_df
from app.strategy.order_blocks import check_ob_mitigation, find_order_blocks


def _in_session(hour: int, session: tuple[int, int]) -> bool:
    start, end = int(session[0]), int(session[1])
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def resolve_killzone(now: datetime, settings: Settings) -> SessionSnapshot:
    if not settings.crt_killzone_enabled:
        return SessionSnapshot(
            killzone_enabled=False,
            killzone_active=True,
            active_killzone=None,
            summary="Killzone filter disabled",
        )

    hour = now.hour
    if _in_session(hour, settings.crt_london_session):
        return SessionSnapshot(
            killzone_enabled=True,
            killzone_active=True,
            active_killzone="london",
            summary="Active London killzone",
        )
    if _in_session(hour, settings.crt_new_york_session):
        return SessionSnapshot(
            killzone_enabled=True,
            killzone_active=True,
            active_killzone="new_york",
            summary="Active New York killzone",
        )
    return SessionSnapshot(
        killzone_enabled=True,
        killzone_active=False,
        active_killzone=None,
        summary="Outside configured killzones",
    )


def _extract_liquidity_levels(ltf_df: pd.DataFrame, settings: Settings) -> tuple[list[float], list[float], float | None, float | None]:
    swings = find_swing_points(ltf_df, settings.swing_lookback)
    levels = find_equal_levels(swings, tolerance=settings.crt_equal_level_tolerance)
    bsl = sorted({float(level.price) for level in levels if level.type == "EQUAL_HIGHS"})
    ssl = sorted({float(level.price) for level in levels if level.type == "EQUAL_LOWS"})

    if not bsl:
        bsl = sorted({float(s.price) for s in swings if s.type in {"HH", "LH"}})
    if not ssl:
        ssl = sorted({float(s.price) for s in swings if s.type in {"HL", "LL"}})

    close_price = float(ltf_df.iloc[-1]["close"])
    bsl_above = [x for x in bsl if x > close_price]
    ssl_below = [x for x in ssl if x < close_price]
    nearest_bsl = min(bsl_above, default=(min(bsl, default=None)))
    nearest_ssl = max(ssl_below, default=(max(ssl, default=None)))
    return bsl, ssl, nearest_bsl, nearest_ssl


def _check_mss_confirmation(ltf_df: pd.DataFrame, settings: Settings, expected_direction: Literal["LONG", "SHORT"]) -> bool:
    swings = find_swing_points(ltf_df, settings.swing_lookback)
    trend = determine_trend(swings)
    choch = detect_choch(ltf_df, swings, trend)
    if not choch:
        return False
    last_break = choch[-1]
    if expected_direction == "LONG" and last_break.direction != "BULLISH":
        return False
    if expected_direction == "SHORT" and last_break.direction != "BEARISH":
        return False
    idx = ltf_df.index.get_indexer([pd.Timestamp(last_break.timestamp)], method="nearest")[0]
    min_idx = max(0, len(ltf_df) - 1 - settings.crt_mss_lookback)
    return idx >= min_idx


def _check_ob_confirmation(ltf_df: pd.DataFrame, settings: Settings, expected_direction: Literal["LONG", "SHORT"]) -> bool:
    swings = find_swing_points(ltf_df, settings.swing_lookback)
    trend = determine_trend(swings)
    structure = detect_bos(ltf_df, swings, trend) + detect_choch(ltf_df, swings, trend)
    direction = "BULLISH" if expected_direction == "LONG" else "BEARISH"
    active_obs = [
        x
        for x in find_order_blocks(ltf_df, structure, max_age=settings.ob_max_age_candles)
        if x.type == direction and not check_ob_mitigation(x, ltf_df)
    ]
    return bool(active_obs)


def _check_fvg_confirmation(ltf_df: pd.DataFrame, settings: Settings, expected_direction: Literal["LONG", "SHORT"]) -> bool:
    direction = "BULLISH" if expected_direction == "LONG" else "BEARISH"
    fvgs = find_fvg(ltf_df, settings.fvg_min_size_percent)
    return any(x.type == direction for x in fvgs)


def _confidence_from_confirmations(confirmations: list[str], killzone_active: bool) -> Literal["HIGH", "MEDIUM", "LOW"]:
    score = len(confirmations) + (1 if killzone_active else 0)
    if score >= 4:
        return "HIGH"
    if score >= 3:
        return "MEDIUM"
    return "LOW"


def analyze_crt_ict_from_df(
    symbol: str,
    ltf_df: pd.DataFrame,
    htf_4h_df: pd.DataFrame,
    htf_1d_df: pd.DataFrame,
    settings: Settings,
    now: datetime | None = None,
) -> tuple[CRTICTAnalysisResponse, TradeSetup | None]:
    if ltf_df.empty or len(ltf_df) < max(30, settings.crt_range_lookback + 2):
        ts = now or datetime.now(timezone.utc)
        empty = CRTICTAnalysisResponse(
            symbol=symbol,
            timeframe=settings.crt_entry_timeframes[0] if settings.crt_entry_timeframes else "5m",
            timestamp=ts,
            liquidity=LiquiditySnapshot(summary="Insufficient candles for liquidity scan"),
            crt=CRTSnapshot(detected=False, summary="Insufficient candles for CRT detection"),
            ict=ICTSnapshot(
                htf_4h_bias="RANGING",
                htf_1d_bias="RANGING",
                htf_aligned=False,
                mss_confirmed=False,
                ob_confirmed=False,
                fvg_confirmed=False,
                summary="Insufficient candles for ICT confirmation",
            ),
            session=resolve_killzone(ts, settings),
            signal="NO_SIGNAL",
            entry=None,
            stop=None,
            targets=[],
            rr=None,
        )
        return empty, None

    timeframe = settings.crt_entry_timeframes[0] if settings.crt_entry_timeframes else "5m"
    ts = now or ltf_df.index[-1].to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    close_price = float(ltf_df.iloc[-1]["close"])

    bsl_levels, ssl_levels, nearest_bsl, nearest_ssl = _extract_liquidity_levels(ltf_df, settings)
    liquidity = LiquiditySnapshot(
        bsl_levels=[round(x, 6) for x in bsl_levels[:8]],
        ssl_levels=[round(x, 6) for x in ssl_levels[:8]],
        nearest_bsl=round(nearest_bsl, 6) if nearest_bsl is not None else None,
        nearest_ssl=round(nearest_ssl, 6) if nearest_ssl is not None else None,
        summary=f"BSL={round(nearest_bsl, 6) if nearest_bsl is not None else '-'} / SSL={round(nearest_ssl, 6) if nearest_ssl is not None else '-'}",
    )

    crt = detect_crt_candle(
        ltf_df,
        lookback=settings.crt_range_lookback,
        min_sweep_pct=settings.crt_min_sweep_pct,
        min_wick_body_ratio=settings.crt_min_wick_body_ratio,
    )
    crt_snapshot = CRTSnapshot(
        detected=crt is not None,
        direction=crt.direction if crt else None,
        range_high=round(crt.range_high, 6) if crt else None,
        range_low=round(crt.range_low, 6) if crt else None,
        sweep_level=round(crt.sweep_level, 6) if crt else None,
        close_inside=bool(crt.close_inside) if crt else False,
        rejection_ratio=round(crt.rejection_ratio, 4) if crt else None,
        summary=(
            f"{crt.direction} CRT at sweep={round(crt.sweep_level, 6)}"
            if crt
            else "No CRT false-break candle on latest bar"
        ),
    )

    htf_4h = analyze_htf_from_df(htf_4h_df) if not htf_4h_df.empty else None
    htf_1d = analyze_htf_from_df(htf_1d_df) if not htf_1d_df.empty else None
    htf_4h_bias = htf_4h.bias.value if htf_4h is not None else "RANGING"
    htf_1d_bias = htf_1d.bias.value if htf_1d is not None else "RANGING"
    htf_aligned = htf_4h_bias == htf_1d_bias and htf_4h_bias in {"BULLISH", "BEARISH"}

    mss_confirmed = False
    ob_confirmed = False
    fvg_confirmed = False
    confirmations: list[str] = []
    expected_direction: Literal["LONG", "SHORT"] | None = crt.direction if crt is not None else None

    if expected_direction is not None and htf_aligned:
        mss_confirmed = _check_mss_confirmation(ltf_df, settings, expected_direction)
        ob_confirmed = _check_ob_confirmation(ltf_df, settings, expected_direction)
        fvg_confirmed = _check_fvg_confirmation(ltf_df, settings, expected_direction)
        if mss_confirmed:
            confirmations.append("MSS")
        if ob_confirmed:
            confirmations.append("OB")
        if fvg_confirmed:
            confirmations.append("FVG")

    ict_snapshot = ICTSnapshot(
        htf_4h_bias=htf_4h_bias,  # type: ignore[arg-type]
        htf_1d_bias=htf_1d_bias,  # type: ignore[arg-type]
        htf_aligned=htf_aligned,
        mss_confirmed=mss_confirmed,
        ob_confirmed=ob_confirmed,
        fvg_confirmed=fvg_confirmed,
        confirmations=confirmations,
        summary=f"HTF aligned={htf_aligned}; confirmations={','.join(confirmations) if confirmations else 'none'}",
    )

    session = resolve_killzone(ts, settings)
    signal: Literal["LONG", "SHORT", "NO_SIGNAL"] = "NO_SIGNAL"
    entry: float | None = None
    stop: float | None = None
    targets: list[float] = []
    rr: float | None = None
    setup: TradeSetup | None = None

    if crt is not None and htf_aligned:
        htf_bias_expected = "BULLISH" if crt.direction == "LONG" else "BEARISH"
        enough_confirmations = len(confirmations) >= 2
        killzone_ok = session.killzone_active or not session.killzone_enabled
        if htf_bias_expected == htf_1d_bias and enough_confirmations and killzone_ok:
            entry = close_price
            buffer = float(settings.crt_stop_buffer_bps) / 10_000
            stop = crt.sweep_level * (1 - buffer) if crt.direction == "LONG" else crt.sweep_level * (1 + buffer)
            risk = abs(entry - stop)
            if risk > 0:
                if crt.direction == "LONG":
                    tp1 = nearest_bsl if nearest_bsl is not None and nearest_bsl > entry else entry + (risk * 2)
                    tp2 = entry + (risk * 3)
                    tp3 = entry + (risk * 4)
                else:
                    tp1 = nearest_ssl if nearest_ssl is not None and nearest_ssl < entry else entry - (risk * 2)
                    tp2 = entry - (risk * 3)
                    tp3 = entry - (risk * 4)
                rr = abs((tp2 - entry) / risk)
                if rr >= settings.crt_min_rr:
                    signal = crt.direction
                    targets = [tp1, tp2, tp3]
                    confidence = _confidence_from_confirmations(confirmations, session.killzone_active)
                    confluences = [f"CRT_{signal}"] + confirmations
                    if session.killzone_active:
                        confluences.append(f"KILLZONE_{session.active_killzone or 'active'}")
                    size = RiskManager(settings).calculate_position_size(entry, stop, settings.risk_per_trade_percent, settings.account_balance_usdt)
                    setup = TradeSetup(
                        timestamp=ts,
                        symbol=symbol,
                        direction=signal,
                        setup_type="CRT_ICT",
                        htf_bias=htf_1d_bias,  # type: ignore[arg-type]
                        entry=float(entry),
                        stop_loss=float(stop),
                        take_profits=[float(tp1), float(tp2), float(tp3)],
                        risk_reward=float(rr),
                        confidence=confidence,
                        confluences=confluences,
                        position_size_usdt=size,
                    )

    analysis = CRTICTAnalysisResponse(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=ts,
        liquidity=liquidity,
        crt=crt_snapshot,
        ict=ict_snapshot,
        session=session,
        signal=signal,
        entry=round(entry, 6) if entry is not None else None,
        stop=round(stop, 6) if stop is not None else None,
        targets=[round(x, 6) for x in targets],
        rr=round(rr, 6) if rr is not None else None,
    )
    return analysis, setup
