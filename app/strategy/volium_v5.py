"""Separate causal interpretation of the second VOLIUM video; v1-v4 stay frozen.

The v1 context proxy is retained, with H1 agreement for intraday. New source
confirmation is engulfment OR opposite FVG inversion plus a recovery fraction.
The pre-raid structural target anchors the entire correction denominator.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from math import isclose, isfinite
from typing import Literal

import numpy as np
import pandas as pd

from app.schemas.setup import TradeSetup
from app.strategy import volium as v1


@dataclass(frozen=True)
class V5Parameters:
    liquidity_mode: Literal["strict", "equal_clusters"] = "strict"
    recovery_fraction: float = 1 / 3
    max_wait_bars: int = 12

    def __post_init__(self):
        if self.liquidity_mode not in {"strict", "equal_clusters"}:
            raise ValueError("Unsupported v5 liquidity interpretation")
        if isinstance(self.recovery_fraction, bool) or not isfinite(self.recovery_fraction) or not isclose(self.recovery_fraction, 1 / 3, rel_tol=0, abs_tol=1e-12):
            raise ValueError("V5 preregistration fixes recovery_fraction to one third")
        if isinstance(self.max_wait_bars, bool) or not isinstance(self.max_wait_bars, int) or self.max_wait_bars != 12:
            raise ValueError("V5 preregistration fixes max_wait_bars to 12")


@dataclass
class V5Diagnostic:
    setup: TradeSetup | None
    reason: str
    features: dict

    def as_dict(self):
        return {"accepted": self.setup is not None,
                "setup": None if self.setup is None else self.setup.model_dump(mode="json"),
                "reason": self.reason, "features": self.features}


@dataclass(frozen=True)
class _Level:
    index: int
    price: float
    kind: str
    known_index: int
    members: tuple[int, ...]


@lru_cache(maxsize=8192)
def _cached_equal_points(indices, prices, lookback):
    """Exact equal touches share confirmation after the last nearby touch.

    A neighbouring equal extreme may be separated by up to ``lookback`` bars,
    provided none crosses the level. This also includes an adjacent plateau.
    More distant strict extrema are combined later by their exact price.
    """
    if not prices:
        return ()
    matrix = np.asarray(prices, dtype=float)
    points = []
    for column, kind in ((1, "high"), (2, "low")):
        values = matrix[:, column]
        start = lookback
        while start + lookback < len(values):
            end = start
            while end + 1 < len(values):
                neighbours = values[end+1:end+lookback+1]
                equal = np.flatnonzero(neighbours == values[start])
                if not len(equal):
                    break
                next_touch = end + 1 + int(equal[0])
                between = values[end+1:next_touch]
                intact = bool((between <= values[start]).all() if kind == "high" else (between >= values[start]).all())
                if not intact:
                    break
                end = next_touch
            if end + lookback < len(values):
                side = np.r_[values[start-lookback:start], values[end+1:end+lookback+1]]
                valid = bool((side < values[start]).all() if kind == "high" else (side > values[start]).all())
                if valid:
                    members = tuple(start + int(i) for i in np.flatnonzero(values[start:end+1] == values[start]))
                    points.append((start, float(values[start]), kind, end + lookback, members))
            start = end + 1
    return tuple(sorted(points))


def _live_levels(frame, lookback, mode, kind):
    if mode == "strict":
        return [_Level(p.index, p.price, p.kind, p.index + lookback, (p.index,))
                for p in v1._unswept(frame, v1._pivots(frame, lookback), kind)]
    points = _cached_equal_points(*v1._immutable_ohlc(frame), lookback)
    values = frame.high.to_numpy(dtype=float) if kind == "high" else frame.low.to_numpy(dtype=float)
    grouped = {}
    for index, price, point_kind, known_index, touches in points:
        if point_kind != kind:
            continue
        after = values[index+1:]
        consumed = bool((after > price).any() if kind == "high" else (after < price).any())
        if not consumed:
            grouped.setdefault(price, []).append((index, known_index, touches))
    result = []
    for price, members in grouped.items():
        result.append(_Level(min(p[0] for p in members), price, kind,
                             max(p[1] for p in members), tuple(sorted({i for p in members for i in p[2]}))))
    return sorted(result, key=lambda p: p.index)


def _structural_target(known, points, direction, prior_price, mode):
    kind = "high" if direction == "LONG" else "low"
    candidates = [p for p in v1._unswept(known, points, kind)
                  if (p.price > prior_price if direction == "LONG" else p.price < prior_price)]
    if not candidates:
        return None
    if mode == "scalp":
        return min(candidates, key=lambda p: p.price) if direction == "LONG" else max(candidates, key=lambda p: p.price)
    return max(candidates, key=lambda p: p.index)


def _entry_model(entry, first, actual, target_open, liquidity_duration, direction, max_wait, entry_duration):
    """Literal final-body engulfment OR an associated, not previously inverted gap."""
    last = len(entry) - 1
    confirm = entry.iloc[last]
    long = direction == "LONG"
    if not (confirm.close > confirm.open if long else confirm.close < confirm.open):
        return None
    changes = entry.close.iloc[actual:].diff().dropna()
    if not bool((changes > 0).all() if long else (changes < 0).all()):
        return None
    anchor_i = None
    for i in range(actual, max(-1, first-max_wait-1), -1):
        row = entry.iloc[i]
        if (row.close < row.open if long else row.close > row.open) and i < last:
            anchor_i = i
            break
    engulf = False
    if anchor_i is not None:
        anchor = entry.iloc[anchor_i]
        engulf = bool(confirm.open <= anchor.close and confirm.close >= anchor.open if long
                      else confirm.open >= anchor.close and confirm.close <= anchor.open)
    start = max(2, int(entry.index.searchsorted(target_open + liquidity_duration, side="left")) + 2)
    inversion = None
    for i in range(start, actual+1):
        if entry.index[i]-entry.index[i-1] != entry_duration or entry.index[i-1]-entry.index[i-2] != entry_duration:
            continue
        older, middle, newer = entry.iloc[i-2], entry.iloc[i-1], entry.iloc[i]
        formed = older.low > newer.high and middle.close < middle.open if long else older.high < newer.low and middle.close > middle.open
        if not formed:
            continue
        far = float(older.low if long else older.high)
        # A weak early inversion during THIS recovery may become valid once the
        # full fraction is reached. An inversion before its actual raid extreme
        # is an old consumed gap and does not supply this event's confirmation.
        closes = entry.close.iloc[i+1:actual+1]
        prior_inversion = bool((closes > far).any() if long else (closes < far).any())
        accepted = confirm.close > far if long else confirm.close < far
        if accepted and not prior_inversion:
            inversion = {"fvg_created_open_utc": entry.index[i].isoformat(),
                         "fvg_known_close_utc": (entry.index[i] + entry_duration).isoformat(),
                         "fvg_far_boundary": far,
                         "fvg_near_boundary": float(newer.high if long else newer.low)}
            break
    if not engulf and inversion is None:
        return None
    return {"confirmation_model": "engulfment_and_inversion" if engulf and inversion else "engulfment" if engulf else "fvg_inversion",
            "engulfed_open_utc": None if anchor_i is None or not engulf else entry.index[anchor_i].isoformat(),
            **(inversion or {})}


def diagnose_volium_v5_from_df(*, symbol, frames, settings=None, mode="intraday", now=None,
                              parameters: V5Parameters | None = None, enforce_session_filter=True):
    parameters = parameters or V5Parameters()
    features = {"version": "experimental_v5", "mode": mode, "parameters": asdict(parameters)}
    def reject(reason):
        return V5Diagnostic(None, reason, features)
    if mode not in {"intraday", "scalp"}:
        raise ValueError("V5 is defined for intraday and the explicitly declared scalp transfer")
    current = pd.Timestamp(now or datetime.now(timezone.utc))
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    context_tf, liquidity_tf, entry_tf = ("1d", "1h", "5m") if mode == "intraday" else ("1h", "5m", "1m")
    if not {context_tf, liquidity_tf, entry_tf}.issubset(frames):
        return reject("missing_frames")
    lookback = int(v1._setting(settings, "volium_swing_lookback", 2))
    context_limit = int(v1._setting(settings, "volium_context_lookback", 80))
    if lookback < 1 or context_limit < 2*lookback + 5:
        raise ValueError("Invalid v5 structural interpretation")
    entry = v1._closed(frames[entry_tf], entry_tf, current)
    if len(entry) < 3:
        return reject("insufficient_entry_history")
    entry_duration = pd.Timedelta(seconds=v1._DURATIONS[entry_tf])
    liquidity_duration = pd.Timedelta(seconds=v1._DURATIONS[liquidity_tf])
    signal_close = entry.index[-1] + entry_duration
    if current - signal_close >= 2*entry_duration:
        return reject("stale_signal")
    timed = enforce_session_filter and v1._setting(settings, "volium_session_enabled", True)
    if timed and (not v1.in_volium_session(signal_close, settings) or not v1.in_volium_session(current, settings)):
        return reject("outside_session")
    context_closed = v1._closed(frames[context_tf], context_tf, current)
    context = context_closed.tail(context_limit)
    direction = v1._trend(v1._pivots(context, lookback))
    if direction is None:
        return reject("no_v1_context_trend")
    features.update(direction=direction, context_method="frozen v1 numeric context")
    price = float(entry.close.iloc[-1])
    liquidity = context_closed if liquidity_tf == context_tf else v1._closed(frames[liquidity_tf], liquidity_tf, current)
    if mode == "intraday":
        if v1._setting(settings, "volium_require_daily_origin_sweep", True) and not v1._origin_sweep(context, direction, lookback):
            return reject("no_v1_daily_origin")
        target_kind = "high" if direction == "LONG" else "low"
        daily_targets = [p for p in v1._unswept(context, v1._pivots(context, lookback), target_kind)
                         if (p.price > price if direction == "LONG" else p.price < price)]
        intact = []
        for p in daily_targets:
            life = context.index[p.index] + pd.Timedelta(days=1)
            consumed = any((frame.loc[frame.index >= life].high.ge(p.price).any() if direction == "LONG"
                            else frame.loc[frame.index >= life].low.le(p.price).any()) for frame in (liquidity, entry))
            if not consumed:
                intact.append(p)
        if not intact:
            return reject("no_unhit_daily_target")
        daily_target = min(intact, key=lambda p: p.price) if direction == "LONG" else max(intact, key=lambda p: p.price)
        features.update(context_target=daily_target.price,
                        context_target_open_utc=context.index[daily_target.index].isoformat(),
                        context_target_known_close_utc=(context.index[daily_target.index]+pd.Timedelta(days=lookback+1)).isoformat())
        h1_direction = v1._trend(v1._pivots(liquidity.tail(context_limit), lookback))
        features["h1_direction"] = h1_direction
        if h1_direction != direction:
            return reject("h1_context_misaligned")
    elif not v1._active_trend(context, direction, float(v1._setting(settings, "volium_active_trend_min_atr", 1.0))):
        return reject("inactive_hourly_trend")
    if liquidity.empty:
        return reject("empty_liquidity_frame")
    matrix = entry[["open", "high", "low", "close"]].to_numpy(dtype=float)
    long = direction == "LONG"
    last = len(entry)-1
    opposite = "low" if long else "high"
    failure = "no_valid_liquidity_raid"
    # Oldest first identifies an event at its first breach, including H1-boundary crossings.
    for first in range(max(1, last-parameters.max_wait_bars), last+1):
        raid_start = entry.index[first]
        last_known = liquidity.index.searchsorted(raid_start-liquidity_duration, side="right")
        known = liquidity.iloc[:last_known].tail(context_limit)
        if known.empty:
            continue
        known_close = known.index[-1] + liquidity_duration
        if entry.index[0] > known_close:
            continue  # Cannot know whether this still-forming liquidity bar already consumed its level.
        levels = [p for p in _live_levels(known, lookback, parameters.liquidity_mode, opposite)
                  if (p.price < matrix[first-1, 3] if long else p.price > matrix[first-1, 3])]
        if not levels:
            continue
        level = max(levels, key=lambda p: p.price) if long else min(levels, key=lambda p: p.price)
        values = matrix[:, 2] if long else matrix[:, 1]
        if not (values[first] < level.price if long else values[first] > level.price):
            continue
        boundary = int(entry.index.searchsorted(known_close, side="left"))
        if boundary >= len(entry) or entry.index[boundary] != known_close or not entry.index[boundary:].to_series().diff().dropna().eq(entry_duration).all():
            continue
        earlier = values[boundary:first]
        if bool((earlier < level.price).any() if long else (earlier > level.price).any()):
            continue
        actual = first + int(np.argmin(values[first:]) if long else np.argmax(values[first:]))
        sweep_time = entry.index[actual]
        if timed and (not v1.in_volium_session(raid_start, settings) or not v1.in_volium_session(sweep_time, settings)):
            continue
        points = v1._pivots(known, lookback)
        target_point = _structural_target(known, points, direction, float(matrix[first-1, 3]), mode)
        if target_point is None:
            failure = "no_preknown_structural_target"
            continue
        target = target_point.price
        after = entry.iloc[first:]
        if (after.high.ge(target).any() if long else after.low.le(target).any()):
            failure = "structural_target_consumed"
            continue
        extreme = float(values[actual])
        span = target-extreme if long else extreme-target
        recovered = price-extreme if long else extreme-price
        fraction = recovered/span if span > 0 else 0.0
        features.update(raid_start_open_utc=raid_start.isoformat(), sweep_open_utc=sweep_time.isoformat(),
                        liquidity_level=level.price, liquidity_level_open_utc=known.index[level.index].isoformat(),
                        liquidity_level_known_close_utc=(known.index[level.known_index]+liquidity_duration).isoformat(),
                        liquidity_pool_members=len(level.members), liquidity_known_close_utc=known_close.isoformat(),
                        raid_extreme=extreme, raid_age_bars=last-first, reaction_age_bars=last-first,
                        recovery_fraction=fraction, manipulation_origin=target, manipulation_span=span,
                        correction_origin_open_utc=known.index[target_point.index].isoformat(),
                        target_known_close_utc=(known.index[target_point.index+lookback]+liquidity_duration).isoformat())
        if span <= 0 or fraction < parameters.recovery_fraction:
            failure = "insufficient_whole_manipulation_recovery"
            continue
        if not (price > level.price if long else price < level.price):
            failure = "raid_not_reclaimed"
            continue
        model = _entry_model(entry, first, actual, known.index[target_point.index], liquidity_duration, direction, parameters.max_wait_bars, entry_duration)
        if model is None:
            failure = "no_direct_engulfment_or_inversion"
            continue
        buffer = float(v1._setting(settings, "volium_scalp_stop_buffer_bps" if mode == "scalp" else "volium_stop_buffer_bps", 10.0 if mode == "scalp" else 2.0))
        if not isfinite(buffer) or not 0 <= buffer < 10000:
            raise ValueError("Invalid v5 stop buffer")
        stop = extreme*(1-buffer/10000) if long else extreme*(1+buffer/10000)
        limit = v1.rr2_entry(stop, target)
        if not (stop < limit <= price < target if long else target < price <= limit < stop):
            failure = "unavailable_2r_retest_geometry"
            continue
        features.update(model, entry=limit, stop=stop, target=target, confirmed_close_utc=signal_close.isoformat())
        identifier = sha256(f"volium-v5:{mode}:{symbol}:{context_tf}:{direction}:{raid_start.isoformat()}".encode()).hexdigest()[:20]
        setup = TradeSetup(id=identifier, timestamp=signal_close.to_pydatetime(), symbol=symbol, direction=direction,
                           setup_type=f"VOLIUM_{mode.upper()}", htf_bias="BULLISH" if long else "BEARISH",
                           entry=limit, stop_loss=stop, take_profits=[target], risk_reward=2, confidence="MEDIUM",
                           confluences=["Experimental v5; declared causal interpretation", features["context_method"],
                                        "D1/H1 aligned" if mode == "intraday" else "Declared M1 transfer of the entry rule",
                                        f"{liquidity_tf} {parameters.liquidity_mode} liquidity raid",
                                        model["confirmation_model"], f"Whole-correction recovery {fraction:.4g}",
                                        "Full known raid extreme stop; structural target; fixed 2R retest"])
        return V5Diagnostic(setup, "accepted", features)
    return reject(failure)


def analyze_volium_v5_from_df(**kwargs):
    return diagnose_volium_v5_from_df(**kwargs).setup


def make_v5_analyzer(parameters: V5Parameters | None = None):
    parameters = parameters or V5Parameters()
    def analyzer(**kwargs):
        return analyze_volium_v5_from_df(parameters=parameters, **kwargs)
    return analyzer


def diagnose_volium_v5_batch_from_df(*, parameter_sets, **kwargs):
    # Only two frozen liquidity interpretations, sharing immutable v1/plateau caches.
    return [diagnose_volium_v5_from_df(parameters=p, **kwargs) for p in parameter_sets]


def analyze_volium_v5_batch_from_df(*, parameter_sets, **kwargs):
    return [result.setup for result in diagnose_volium_v5_batch_from_df(parameter_sets=parameter_sets, **kwargs)]
