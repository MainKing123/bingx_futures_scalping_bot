"""Deterministic interpretation of the three liquidity setups in the source video.

Inputs are OHLC frames indexed by candle OPEN time. No network or order calls.
See docs/strategy.md for the distinction between source rules and numeric choices.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from functools import lru_cache
from math import isfinite
from typing import Literal

import numpy as np
import pandas as pd

from app.schemas.setup import TradeSetup

Mode = Literal["intraday", "scalp", "swing"]
_DURATIONS = {"1m": 60, "5m": 300, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}


@dataclass(frozen=True)
class _Pivot:
    index: int
    price: float
    kind: str


def _setting(settings: object | None, name: str, default):
    return getattr(settings, name, default)


def _closed(frame: pd.DataFrame, timeframe: str, now: pd.Timestamp) -> pd.DataFrame:
    """Reject corrupt OHLC and exclude forming/future candles before any calculation."""
    if frame is None or frame.empty or not {"open", "high", "low", "close"}.issubset(frame):
        return pd.DataFrame()
    try:
        df = frame
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None or str(df.index.tz) != "UTC":
            df = df.copy()
            df.index = pd.to_datetime(df.index, utc=True)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if df.index.has_duplicates:
            return pd.DataFrame()
        last = df.index.searchsorted(now - pd.Timedelta(seconds=_DURATIONS[timeframe]), side="right")
        df = df.iloc[:last]
        columns = ["open", "high", "low", "close"]
        if not all(pd.api.types.is_numeric_dtype(df[col].dtype) for col in columns):
            df = df.copy()
            for col in columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        values = df[columns].to_numpy(dtype=float, copy=False)
        if not np.isfinite(values).all():
            return pd.DataFrame()
        opens, highs, lows, closes = values.T
        valid = (lows > 0) & (lows <= np.minimum(opens, closes))
        valid &= (highs >= np.maximum(opens, closes)) & (highs >= lows)
        return df if valid.all() else pd.DataFrame()
    except (ValueError, TypeError, OverflowError):
        return pd.DataFrame()


def _immutable_ohlc(df: pd.DataFrame) -> tuple[tuple, tuple]:
    """Value keys, never frame identities: mutation/updated candles invalidate cache."""
    indices = tuple(df.index.asi8) if isinstance(df.index, pd.DatetimeIndex) else tuple(df.index)
    prices = tuple(map(tuple, df[["open", "high", "low", "close"]].to_numpy(dtype=float, copy=False)))
    return indices, prices


@lru_cache(maxsize=8192)
def _cached_pivots(indices: tuple, prices: tuple, lookback: int) -> tuple[_Pivot, ...]:
    result = []
    if not prices:
        return ()
    values = np.asarray(prices, dtype=float)
    highs, lows = values[:, 1], values[:, 2]
    for i in range(lookback, len(prices) - lookback):
        high, low = float(highs[i]), float(lows[i])
        if high > max(float(highs[i - lookback:i].max()), float(highs[i + 1:i + lookback + 1].max())):
            result.append(_Pivot(i, high, "high"))
        if low < min(float(lows[i - lookback:i].min()), float(lows[i + 1:i + lookback + 1].min())):
            result.append(_Pivot(i, low, "low"))
    return tuple(result)


def _pivots(df: pd.DataFrame, lookback: int) -> list[_Pivot]:
    """A pivot is known only after lookback subsequent CLOSED candles exist."""
    return list(_cached_pivots(*_immutable_ohlc(df), lookback))


def _trend(points: list[_Pivot]) -> str | None:
    highs = [p.price for p in points if p.kind == "high"]
    lows = [p.price for p in points if p.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return None
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return "LONG"
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return "SHORT"
    return None


def _unswept(df: pd.DataFrame, points: list[_Pivot], kind: str) -> list[_Pivot]:
    result = []
    values = df.high.to_numpy(dtype=float) if kind == "high" else df.low.to_numpy(dtype=float)
    for point in points:
        if point.kind != kind:
            continue
        later = values[point.index + 1:]
        touched = (later >= point.price).any() if kind == "high" else (later <= point.price).any()
        if not touched:
            result.append(point)
    return result


def _target(df: pd.DataFrame, points: list[_Pivot], direction: str, price: float) -> float | None:
    kind = "high" if direction == "LONG" else "low"
    levels = [p.price for p in _unswept(df, points, kind)]
    levels = [p for p in levels if p > price] if direction == "LONG" else [p for p in levels if p < price]
    return (min(levels) if direction == "LONG" else max(levels)) if levels else None


def _origin_sweep(df: pd.DataFrame, direction: str, lookback: int) -> bool:
    """Conservative daily point A: prior confirmed opposite pivot swept/reclaimed."""
    return _cached_origin_sweep(*_immutable_ohlc(df), direction, lookback)


@lru_cache(maxsize=2048)
def _cached_origin_sweep(indices: tuple, prices: tuple, direction: str, lookback: int) -> bool:
    if not prices:
        return False
    kind = "low" if direction == "LONG" else "high"
    matrix = np.asarray(prices, dtype=float)
    lows, highs, closes = matrix[:, 2], matrix[:, 1], matrix[:, 3]
    values = lows if direction == "LONG" else highs
    suffix = np.minimum.accumulate(lows[::-1])[::-1] if direction == "LONG" else np.maximum.accumulate(highs[::-1])[::-1]
    # The earliest touch consumes a level. Its pivot must have been confirmed
    # strictly BEFORE that touch; precomputing pivots adds no future information.
    for point in _cached_pivots(indices, prices, lookback):
        if point.kind != kind:
            continue
        later = values[point.index + 1:]
        touched = np.flatnonzero(later <= point.price if direction == "LONG" else later >= point.price)
        if not len(touched):
            continue
        i = point.index + 1 + int(touched[0])
        if i < 2 * lookback + 2 or point.index + lookback >= i:
            continue
        if direction == "LONG" and lows[i] < point.price < closes[i] and suffix[i] >= lows[i]:
            return True
        if direction == "SHORT" and highs[i] > point.price > closes[i] and suffix[i] <= highs[i]:
            return True
    return False


def _active_trend(df: pd.DataFrame, direction: str, min_atr: float) -> bool:
    return _cached_active_trend(*_immutable_ohlc(df), direction, min_atr)


@lru_cache(maxsize=2048)
def _cached_active_trend(indices: tuple, prices: tuple, direction: str, min_atr: float) -> bool:
    df = pd.DataFrame(prices, columns=["open", "high", "low", "close"])
    if len(df) < 5:
        return False
    prev = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    move = float(df.close.iloc[-1] - df.close.iloc[-4])
    return atr > 0 and (move if direction == "LONG" else -move) >= atr * min_atr


def is_volium_session(timestamp: datetime | pd.Timestamp, windows: list[tuple[str, str]]) -> bool:
    """Fixed UTC+3 used in the video; boundaries are start-inclusive/end-exclusive."""
    moment = pd.Timestamp(timestamp)
    moment = moment.tz_localize("UTC") if moment.tzinfo is None else moment.tz_convert("UTC")
    local = moment.to_pydatetime().astimezone(timezone(timedelta(hours=3)))
    minute = local.hour * 60 + local.minute
    return _minute_in_windows(minute, windows)


def _minute_in_windows(minute: int, windows: list[tuple[str, str]]) -> bool:
    for start, end in windows:
        start_h, start_m = map(int, start.split(":"))
        end_h, end_m = map(int, end.split(":"))
        if not (0 <= start_h <= 23 and 0 <= end_h <= 23 and 0 <= start_m <= 59 and 0 <= end_m <= 59):
            raise ValueError("Session windows must use HH:MM")
        a, b = start_h * 60 + start_m, end_h * 60 + end_m
        if a < b and a <= minute < b or a > b and (minute >= a or minute < b):
            return True
    return False


def in_volium_session(timestamp: datetime | pd.Timestamp, settings: object | None = None) -> bool:
    """Fixed video UTC+3 table by default; optional DST-aware market-local hypothesis."""
    if _setting(settings, "volium_session_clock", "fixed_utc3") == "fixed_utc3":
        return is_volium_session(timestamp, _setting(settings, "volium_sessions_utc3", [("10:00", "12:00"), ("16:30", "18:00")]))
    moment = pd.Timestamp(timestamp)
    moment = moment.tz_localize("UTC") if moment.tzinfo is None else moment.tz_convert("UTC")
    markets = _setting(settings, "volium_market_sessions", [
        ("Europe/London", "08:00", "10:00"), ("America/New_York", "09:30", "11:00")])
    for zone, start, end in markets:
        local = moment.tz_convert(zone)
        if _minute_in_windows(local.hour * 60 + local.minute, [(start, end)]):
            return True
    return False


def rr2_entry(stop_loss: float, take_profit: float) -> float:
    """Solve fixed structural TP/SL for reward:risk = 2:1 without moving either."""
    if not all(isfinite(x) and x > 0 for x in (stop_loss, take_profit)) or stop_loss == take_profit:
        raise ValueError("Distinct finite positive stop and target are required")
    return (take_profit + 2 * stop_loss) / 3


def _reaction(df: pd.DataFrame, sweep_i: int, level: float, direction: str, max_bars: int, min_body: float) -> bool:
    """Numeric V interpretation: monotone closes, no resweep, directional body clearance."""
    last = len(df) - 1
    if last - sweep_i > max_bars:
        return False
    sweep, confirm = df.iloc[sweep_i], df.iloc[last]
    bars = df.iloc[sweep_i:last + 1]
    candle_range = float(confirm.high - confirm.low)
    if candle_range <= 0 or abs(float(confirm.close - confirm.open)) / candle_range < min_body:
        return False
    anchor = None
    for i in range(sweep_i, max(-1, sweep_i - max_bars - 1), -1):
        row = df.iloc[i]
        opposite = row.close < row.open if direction == "LONG" else row.close > row.open
        if opposite and i < last:
            anchor = row
            break
    if anchor is None:
        return False
    changes = bars.close.diff().dropna()
    if direction == "LONG":
        return bool(confirm.close > confirm.open and confirm.close > level and confirm.close >= anchor.open
                    and bars.open.min() <= anchor.close and (changes > 0).all()
                    and bars.low.iloc[1:].ge(sweep.low).all())
    return bool(confirm.close < confirm.open and confirm.close < level and confirm.close <= anchor.open
                and bars.open.max() >= anchor.close and (changes < 0).all()
                and bars.high.iloc[1:].le(sweep.high).all())


def analyze_volium_from_df(
    *, symbol: str, frames: dict[str, pd.DataFrame], settings: object | None = None,
    mode: Mode = "intraday", now: datetime | None = None, enforce_session_filter: bool = True,
) -> TradeSetup | None:
    """Return a new closed-candle signal with a fixed 2R RETEST LIMIT entry, or None.

    The scanner must execute the entry as a limit, never as an immediate market buy.
    Mode uses D1/H1/M5, H1/M5/M1, or D1/H1 (W1/H4 optional) respectively.
    """
    if mode not in {"intraday", "scalp", "swing"}:
        raise ValueError("Unsupported VOLIUM mode")
    current = pd.Timestamp(now or datetime.now(timezone.utc))
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    weekly = mode == "swing" and _setting(settings, "volium_swing_context", "1d") == "1w"
    context_tf, liquidity_tf, entry_tf = (
        ("1d", "1h", "5m") if mode == "intraday" else
        ("1h", "5m", "1m") if mode == "scalp" else
        ("1w", "1w", "4h") if weekly else ("1d", "1d", "1h")
    )
    required = {context_tf, liquidity_tf, entry_tf}
    if not required.issubset(frames):
        return None
    lookback = int(_setting(settings, "volium_swing_lookback", 2))
    context_limit = int(_setting(settings, "volium_context_lookback", 80))
    sweep_age = int(_setting(settings, "volium_sweep_max_age_bars", 12))
    reaction_bars = int(_setting(settings, "volium_reaction_max_bars", 3))
    min_body = float(_setting(settings, "volium_min_body_ratio", 0.6))
    if lookback < 1 or context_limit < 2 * lookback + 5 or sweep_age < 1 or reaction_bars < 0 or not 0 <= min_body <= 1:
        raise ValueError("Invalid VOLIUM interpretation parameters")
    ltf = _closed(frames[entry_tf], entry_tf, current)
    if len(ltf) < 3:
        return None
    # This is necessary for every possible reaction, independent of context.
    # Reject it once before loading context or searching impossible sweeps.
    confirm = ltf.iloc[-1]
    candle_range = float(confirm.high - confirm.low)
    if candle_range <= 0 or abs(float(confirm.close - confirm.open)) / candle_range < min_body:
        return None
    session_required = mode != "swing" and enforce_session_filter and _setting(settings, "volium_session_enabled", True)
    signal_close = ltf.index[-1] + pd.Timedelta(seconds=_DURATIONS[entry_tf])
    if current - signal_close >= pd.Timedelta(seconds=2 * _DURATIONS[entry_tf]):
        return None
    if session_required and (not in_volium_session(signal_close, settings) or not in_volium_session(current, settings)):
        return None
    context_closed = _closed(frames[context_tf], context_tf, current)
    context = context_closed.tail(context_limit)
    if len(context) < 2 * lookback + 5:
        return None
    context_points = _pivots(context, lookback)
    direction = _trend(context_points)
    if direction is None:
        return None
    price = float(ltf.close.iloc[-1])
    context_target = _target(context, context_points, direction, price) if mode != "scalp" else None
    if mode != "scalp" and context_target is None:
        return None
    if mode == "intraday" and _setting(settings, "volium_require_daily_origin_sweep", True):
        if not _origin_sweep(context, direction, lookback):
            return None
    if mode == "scalp" and not _active_trend(context, direction, float(_setting(settings, "volium_active_trend_min_atr", 1.0))):
        return None
    liquidity = context_closed if liquidity_tf == context_tf else _closed(frames[liquidity_tf], liquidity_tf, current)
    if liquidity.empty:
        return None
    # Older candidates cannot pass _reaction's existing maximum age anyway.
    effective_age = min(sweep_age, reaction_bars + 1)
    liquidity_duration = pd.Timedelta(seconds=_DURATIONS[liquidity_tf])
    for sweep_i in range(len(ltf) - 1, max(0, len(ltf) - effective_age - 1), -1):
        sweep_time = ltf.index[sweep_i]
        if session_required and not in_volium_session(sweep_time, settings):
            continue
        last_known = liquidity.index.searchsorted(sweep_time - liquidity_duration, side="right")
        known = liquidity.iloc[:last_known].tail(context_limit)
        points = _pivots(known, lookback)
        prior_price = float(ltf.close.iloc[sweep_i - 1])
        opposite = "low" if direction == "LONG" else "high"
        levels = [p.price for p in _unswept(known, points, opposite)]
        levels = [x for x in levels if x < prior_price] if direction == "LONG" else [x for x in levels if x > prior_price]
        if not levels:
            continue
        level = max(levels) if direction == "LONG" else min(levels)
        sweep = ltf.iloc[sweep_i]
        swept = sweep.low < level if direction == "LONG" else sweep.high > level
        if not swept or not _reaction(ltf, sweep_i, level, direction, reaction_bars, min_body):
            continue
        if mode == "scalp":
            target = _target(known, points, direction, prior_price)
            context_target = target
        else:
            structural = [p for p in points if p.kind == ("high" if direction == "LONG" else "low")]
            target = structural[-1].price if structural else None
        if target is None:
            continue
        # Targets must remain untouched between the sweep and the signal.
        after = ltf.iloc[sweep_i:]
        if direction == "LONG" and (target <= price or float(after.high.max()) >= target):
            continue
        if direction == "SHORT" and (target >= price or float(after.low.min()) <= target):
            continue
        buffer = float(_setting(settings, "volium_scalp_stop_buffer_bps" if mode == "scalp" else "volium_stop_buffer_bps", 10.0 if mode == "scalp" else 2.0))
        if not isfinite(buffer) or not 0 <= buffer < 10000:
            raise ValueError("Stop buffer must be finite and between 0 and 10000 bps")
        stop = float(sweep.low) * (1 - buffer / 10000) if direction == "LONG" else float(sweep.high) * (1 + buffer / 10000)
        entry = rr2_entry(stop, target)
        # A pullback limit is only valid on the already-confirmed side of entry.
        if direction == "LONG" and not stop < entry <= price < target:
            continue
        if direction == "SHORT" and not target < price <= entry < stop:
            continue
        identifier = sha256(f"volium:{mode}:{symbol}:{context_tf}:{sweep_time.isoformat()}".encode()).hexdigest()[:20]
        return TradeSetup(
            id=identifier, timestamp=signal_close.to_pydatetime(), symbol=symbol, direction=direction,
            setup_type=f"VOLIUM_{mode.upper()}", htf_bias="BULLISH" if direction == "LONG" else "BEARISH",
            entry=entry, stop_loss=stop, take_profits=[target], risk_reward=2.0, confidence="MEDIUM",
            confluences=[f"{context_tf} confirmed swing trend", f"{context_tf} unhit liquidity target {context_target:g}",
                         f"{liquidity_tf} nearest opposite liquidity sweep {level:g}",
                         f"{entry_tf} closed aggressive body clearance", "Fixed structural target and sweep stop",
                         "2R retest limit; numeric interpretation of the video",
                         f"Configured session: {_setting(settings, 'volium_session_clock', 'fixed_utc3')}" if session_required else "No session filter (swing or explicitly disabled)"],
        )
    return None
