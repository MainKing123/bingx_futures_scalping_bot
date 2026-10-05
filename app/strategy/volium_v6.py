"""V6: a current liquidity-to-liquidity leg, with an independent M5 origin.

No prior HH/HL trend, future right-hand confirmation of A, or profit fitting.
Numerical choices are disclosed in docs/strategy_v6.md; V1-V5 stay frozen.
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
from app.strategy.volium_v5 import _entry_model, _live_levels, _structural_target


@dataclass(frozen=True)
class V6Parameters:
    liquidity_mode: Literal["equal_clusters"] = "equal_clusters"
    recovery_fraction: float = 1 / 3
    max_wait_bars: int = 12

    def __post_init__(self):
        if self.liquidity_mode != "equal_clusters":
            raise ValueError("V6 fixes one equal-liquidity interpretation")
        if isinstance(self.recovery_fraction, bool) or not isfinite(self.recovery_fraction) or not isclose(self.recovery_fraction, 1 / 3, rel_tol=0, abs_tol=1e-12):
            raise ValueError("V6 fixes recovery_fraction to one third")
        if isinstance(self.max_wait_bars, bool) or not isinstance(self.max_wait_bars, int) or self.max_wait_bars != 12:
            raise ValueError("V6 fixes the reaction wait to 12 bars")


@dataclass
class V6Diagnostic:
    setup: TradeSetup | None
    reason: str
    features: dict

    def as_dict(self):
        return {"accepted": self.setup is not None,
                "setup": None if self.setup is None else self.setup.model_dump(mode="json"),
                "reason": self.reason, "features": self.features}


@dataclass(frozen=True)
class LiquidityLeg:
    direction: str
    first_raid_open_utc: str
    actual_extreme_open_utc: str
    extreme: float
    swept_level: float
    swept_level_known_close_utc: str
    reaction_close_utc: str
    confirmation_model: str
    target: float | None
    target_known_close_utc: str | None
    last_checked_close_utc: str


def manipulation_origin(known_entry, direction, lookback=2):
    """Latest live, already confirmed M5 peak/trough before the first raid.

    Its entire incoming price leg, including the pre-level descent/ascent, is
    measured. It is not the raid wick or the independent H1 take-profit.
    """
    kind = "high" if direction == "LONG" else "low"
    prior_price = float(known_entry.close.iloc[-1]) if len(known_entry) else 0
    points = [p for p in _live_levels(known_entry, lookback, "equal_clusters", kind)
              if (p.price > prior_price if direction == "LONG" else p.price < prior_price)]
    return max(points, key=lambda p: p.index) if points else None


def current_liquidity_leg(frame, timeframe, lookback=2):
    if frame.empty:
        return None
    duration = pd.Timedelta(seconds=v1._DURATIONS[timeframe])
    if not frame.index.to_series().diff().dropna().eq(duration).all():
        return None
    return _cached_liquidity_leg(*v1._immutable_ohlc(frame), timeframe, lookback)


@lru_cache(maxsize=8192)
def _cached_liquidity_leg(indices, prices, timeframe, lookback):
    frame = pd.DataFrame(prices, columns=["open", "high", "low", "close"],
                         index=pd.to_datetime(indices, utc=True))
    duration = pd.Timedelta(seconds=v1._DURATIONS[timeframe])
    require_target = timeframe == "1d"
    confirmed = []
    for first in range(2 * lookback + 2, len(frame)):
        # Every pool is constructed from its actual pre-raid prefix. A future
        # equal touch cannot postpone/hide an earlier pool's confirmation.
        before = frame.iloc[:first]
        prior = float(before.close.iloc[-1])
        directions = []
        for direction, kind in (("LONG", "low"), ("SHORT", "high")):
            levels = [p for p in _live_levels(before, lookback, "equal_clusters", kind)
                      if (p.price < prior if direction == "LONG" else p.price > prior)]
            if not levels:
                continue
            level = max(levels, key=lambda p: p.price) if direction == "LONG" else min(levels, key=lambda p: p.price)
            row = frame.iloc[first]
            breached = row.low < level.price if direction == "LONG" else row.high > level.price
            if breached:
                directions.append((direction, level))
        if len(directions) != 1:
            continue  # An outside bar raiding both sides is deliberately ambiguous.
        direction, level = directions[0]
        long = direction == "LONG"
        opposite_kind = "high" if long else "low"
        prior_targets = [p for p in _live_levels(before, lookback, "equal_clusters", opposite_kind)
                         if (p.price > prior if long else p.price < prior)]
        if require_target and not prior_targets:
            continue
        prior_target = (min(prior_targets, key=lambda p: p.price) if long else max(prior_targets, key=lambda p: p.price)) if prior_targets else None
        for reaction in range(first, min(len(frame), first + 13)):
            prefix = frame.iloc[:reaction + 1]
            value = prefix.low if long else prefix.high
            actual = first + int(np.argmin(value.iloc[first:]) if long else np.argmax(value.iloc[first:]))
            extreme = float(value.iloc[actual])
            price = float(prefix.close.iloc[-1])
            if not (price > level.price if long else price < level.price):
                continue
            model = _entry_model(prefix, first, actual, before.index[level.index], duration,
                                 direction, 12, duration)
            if model is None:
                continue
            # Initial B is known before A. A wick-only visit completes that leg;
            # a close through it permits continuation to the next known pool.
            block = prefix.iloc[first:]
            visits = (block.high.ge(prior_target.price) if long else block.low.le(prior_target.price)) if prior_target else pd.Series(False, index=block.index)
            through = (block.close.gt(prior_target.price) if long else block.close.lt(prior_target.price)) if prior_target else pd.Series(False, index=block.index)
            if require_target and bool((visits & ~through).any()):
                confirmed.append((first, None))
                break
            targets = [p for p in _live_levels(prefix, lookback, "equal_clusters", opposite_kind)
                       if (p.price > price if long else p.price < price)]
            target = min(targets, key=lambda p: p.price) if long and targets else max(targets, key=lambda p: p.price) if targets else None
            if require_target and target is None:
                confirmed.append((first, None))
                break
            target_price = target.price if target and require_target else None
            target_known = (prefix.index[target.known_index] + duration).isoformat() if target and require_target else None
            intact = True
            for i in range(reaction + 1, len(frame)):
                row = frame.iloc[i]
                if row.low < extreme if long else row.high > extreme:
                    intact = False
                    break
                if require_target and (row.high >= target_price if long else row.low <= target_price):
                    if not (row.close > target_price if long else row.close < target_price):
                        intact = False
                        break
                    following = frame.iloc[:i + 1]
                    choices = [p for p in _live_levels(following, lookback, "equal_clusters", opposite_kind)
                               if (p.price > row.close if long else p.price < row.close)]
                    if not choices:
                        intact = False
                        break
                    next_target = min(choices, key=lambda p: p.price) if long else max(choices, key=lambda p: p.price)
                    target_price = next_target.price
                    target_known = (following.index[next_target.known_index] + duration).isoformat()
            leg = LiquidityLeg(direction, frame.index[first].isoformat(), frame.index[actual].isoformat(),
                extreme, level.price, (before.index[level.known_index] + duration).isoformat(),
                (frame.index[reaction] + duration).isoformat(), model["confirmation_model"],
                target_price, target_known, (frame.index[-1] + duration).isoformat()) if intact else None
            confirmed.append((first, leg))
            break
    # A newer completed/invalidated confirmed leg cannot resurrect an old leg.
    return max(confirmed, key=lambda item: item[0])[1] if confirmed else None


def diagnose_volium_v6_from_df(*, symbol, frames, settings=None, mode="intraday", now=None,
                              parameters: V6Parameters | None = None, enforce_session_filter=True):
    parameters = parameters or V6Parameters()
    features = {"version": "experimental_v6", "mode": mode, "parameters": asdict(parameters)}
    def reject(reason):
        return V6Diagnostic(None, reason, features)
    if mode != "intraday":
        raise ValueError("V6 is defined only for the D1/H1/M5 intraday source model")
    current = pd.Timestamp(now or datetime.now(timezone.utc))
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    context_tf, liquidity_tf, entry_tf = ("1d", "1h", "5m") if mode == "intraday" else ("1h", "5m", "1m")
    if not {context_tf, liquidity_tf, entry_tf}.issubset(frames):
        return reject("missing_frames")
    lookback = int(v1._setting(settings, "volium_swing_lookback", 2))
    context_limit = int(v1._setting(settings, "volium_context_lookback", 80))
    if lookback < 1 or context_limit < 2*lookback + 5:
        raise ValueError("Invalid v6 structural interpretation")
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
    liquidity = v1._closed(frames[liquidity_tf], liquidity_tf, current)
    daily_leg = current_liquidity_leg(context_closed.tail(context_limit), context_tf, lookback)
    hourly_leg = current_liquidity_leg(liquidity.tail(context_limit), liquidity_tf, lookback)
    if daily_leg is None:
        return reject("no_current_daily_leg")
    direction = daily_leg.direction
    features.update(direction=direction, context_method="current confirmed A-to-B liquidity leg",
                    daily_leg=asdict(daily_leg), hourly_leg=None if hourly_leg is None else asdict(hourly_leg))
    price = float(entry.close.iloc[-1])
    daily_observed = pd.concat([liquidity, entry])
    for leg, timeframe, lower in ((daily_leg, context_tf, daily_observed), (hourly_leg, liquidity_tf, entry)):
        if leg is None:
            continue
        # Fully closed higher bars have already checked the lifetime. Check the
        # current unclosed higher bar using its observed lower bars as well.
        observed = lower.loc[lower.index >= pd.Timestamp(leg.last_checked_close_utc)]
        if observed.empty:
            continue
        broken = observed.low.lt(leg.extreme).any() if leg.direction == "LONG" else observed.high.gt(leg.extreme).any()
        hit = (observed.high.ge(leg.target).any() if direction == "LONG" else observed.low.le(leg.target).any()) if timeframe == "1d" else False
        if broken or hit:
            return reject("current_context_extreme_broken" if broken else "current_context_target_reached")
    if hourly_leg is None or hourly_leg.direction != direction:
        return reject("hourly_liquidity_leg_misaligned")
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
        origin = manipulation_origin(entry.iloc[:first], direction, lookback)
        if origin is None:
            failure = "no_preknown_m5_manipulation_origin"
            continue
        span = origin.price-extreme if long else extreme-origin.price
        recovered = price-extreme if long else extreme-price
        fraction = recovered/span if span > 0 else 0.0
        features.update(raid_start_open_utc=raid_start.isoformat(), sweep_open_utc=sweep_time.isoformat(),
                        liquidity_level=level.price, liquidity_level_open_utc=known.index[level.index].isoformat(),
                        liquidity_level_known_close_utc=(known.index[level.known_index]+liquidity_duration).isoformat(),
                        liquidity_pool_members=len(level.members), liquidity_known_close_utc=known_close.isoformat(),
                        raid_extreme=extreme, raid_age_bars=last-first, reaction_age_bars=last-first,
                        recovery_fraction=fraction, manipulation_origin=origin.price, manipulation_span=span,
                        manipulation_origin_open_utc=entry.index[origin.index].isoformat(),
                        manipulation_origin_known_close_utc=(entry.index[origin.known_index]+entry_duration).isoformat(),
                        correction_origin_open_utc=known.index[target_point.index].isoformat(),
                        target_known_close_utc=(known.index[target_point.index+lookback]+liquidity_duration).isoformat())
        if span <= 0 or fraction < parameters.recovery_fraction:
            failure = "insufficient_whole_manipulation_recovery"
            continue
        if not (price > level.price if long else price < level.price):
            failure = "raid_not_reclaimed"
            continue
        model = _entry_model(entry, first, actual, entry.index[origin.index]-entry_duration, entry_duration, direction, parameters.max_wait_bars, entry_duration)
        if model is None:
            failure = "no_direct_engulfment_or_inversion"
            continue
        buffer = float(v1._setting(settings, "volium_scalp_stop_buffer_bps" if mode == "scalp" else "volium_stop_buffer_bps", 10.0 if mode == "scalp" else 2.0))
        if not isfinite(buffer) or not 0 <= buffer < 10000:
            raise ValueError("Invalid v6 stop buffer")
        stop = extreme*(1-buffer/10000) if long else extreme*(1+buffer/10000)
        limit = v1.rr2_entry(stop, target)
        if not (stop < limit <= price < target if long else target < price <= limit < stop):
            failure = "unavailable_2r_retest_geometry"
            continue
        features.update(model, entry=limit, stop=stop, target=target, confirmed_close_utc=signal_close.isoformat())
        identifier = sha256(f"volium-v6:{mode}:{symbol}:{context_tf}:{direction}:{raid_start.isoformat()}".encode()).hexdigest()[:20]
        setup = TradeSetup(id=identifier, timestamp=signal_close.to_pydatetime(), symbol=symbol, direction=direction,
                           setup_type=f"VOLIUM_{mode.upper()}", htf_bias="BULLISH" if long else "BEARISH",
                           entry=limit, stop_loss=stop, take_profits=[target], risk_reward=2, confidence="MEDIUM",
                           confluences=["Experimental v6; current liquidity-to-liquidity context", features["context_method"],
                                        "D1/H1 current liquidity legs aligned",
                                        f"{liquidity_tf} {parameters.liquidity_mode} liquidity raid",
                                        model["confirmation_model"], f"Whole-correction recovery {fraction:.4g}",
                                        "Full known raid extreme stop; structural target; fixed 2R retest"])
        return V6Diagnostic(setup, "accepted", features)
    return reject(failure)


def analyze_volium_v6_from_df(**kwargs):
    return diagnose_volium_v6_from_df(**kwargs).setup


def make_v6_analyzer(parameters: V6Parameters | None = None):
    parameters = parameters or V6Parameters()
    def analyzer(**kwargs):
        return analyze_volium_v6_from_df(parameters=parameters, **kwargs)
    return analyzer


def diagnose_volium_v6_batch_from_df(*, parameter_sets, **kwargs):
    # One fixed interpretation; immutable prefix caches are shared across calls.
    return [diagnose_volium_v6_from_df(parameters=p, **kwargs) for p in parameter_sets]


def analyze_volium_v6_batch_from_df(*, parameter_sets, **kwargs):
    return [result.setup for result in diagnose_volium_v6_batch_from_df(parameter_sets=parameter_sets, **kwargs)]
