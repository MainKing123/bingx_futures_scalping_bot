"""One experimental causal post-sweep D1-break hypothesis.

The immediate next-day confirmation is an explicit research interpretation;
frozen v1/v2/v3 and their published research results remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from datetime import datetime,timezone
from functools import lru_cache
from hashlib import sha256
from math import isfinite
from typing import Literal

import numpy as np
import pandas as pd

from app.schemas.setup import TradeSetup
from app.strategy import volium as v1
from app.strategy import volium_v3 as frozen_v3

@dataclass(frozen=True)
class V4Parameters:
    context_mode: Literal["post_sweep_break"] = "post_sweep_break"
    correction_target_mode: Literal["leg_origin"] = "leg_origin"
    reaction_min_atr: float = .8
    reaction_min_prior_body_ratio: float = 1.0
    reaction_atr_period: int = 14
    reaction_min_body_ratio: float = .6
    reaction_max_bars: int = 3

    def __post_init__(self):
        if self.context_mode not in {"post_sweep_break"}:
            raise ValueError("Unsupported v4 context hypothesis")
        if self.correction_target_mode not in {"leg_origin"}:
            raise ValueError("Unsupported v4 correction hypothesis")
        if not all(isfinite(value) for value in (self.reaction_min_atr,self.reaction_min_prior_body_ratio,self.reaction_min_body_ratio)):
            raise ValueError("V4 thresholds must be finite")
        if not 0 <= self.reaction_min_atr <= 10 or not 0 <= self.reaction_min_prior_body_ratio <= 10:
            raise ValueError("V4 strength ratios must be between zero and ten")
        if any(isinstance(value,bool) or not isinstance(value,int) for value in (self.reaction_atr_period,self.reaction_max_bars)):
            raise ValueError("V4 candle counts must be integers")
        if not 0 < self.reaction_min_body_ratio <= 1 or not 2 <= self.reaction_atr_period <= 100 or not 1 <= self.reaction_max_bars <= 20:
            raise ValueError("Invalid v4 candle interpretation parameters")


@dataclass
class V4Diagnostic:
    setup: TradeSetup | None
    reason: str
    features: dict

    def as_dict(self):
        return {"accepted":self.setup is not None,"reason":self.reason,"features":self.features,
                "setup":self.setup.model_dump(mode="json") if self.setup is not None else None}


# Explicit immutable helpers; no module-global substitution or monkeypatch.
_Leg = frozen_v3._Leg
_reaction_features = frozen_v3._reaction_features
_correction_origin = frozen_v3._correction_origin
_current_raid = frozen_v3._current_raid
_raid_reaction = frozen_v3._raid_reaction


@lru_cache(maxsize=2048)
def _cached_post_sweep_break_leg(indices,prices,lookback):
    """Latest actual closed D1 raid, confirmed only by its immediate next day.

    A does not require a trend before it or future pivot-right bars. Its swept
    level must already have been confirmed before A opened. No older event is
    substituted if the newest reclaim is ambiguous or lacks the required C.
    """
    if len(prices) < 2*lookback+2:
        return _Leg("insufficient_context")
    matrix = np.asarray(prices,dtype=float)
    highs,lows,closes = matrix[:,1],matrix[:,2],matrix[:,3]
    points = v1._cached_pivots(indices,prices,lookback)
    events = []
    for i in range(2*lookback+1,len(prices)):
        known = [point for point in points if point.index+lookback < i]
        day_events = []
        for direction in ("LONG","SHORT"):
            long = direction == "LONG"
            kind = "low" if long else "high"
            candidates = []
            for point in known:
                if point.kind != kind or not (point.price < closes[i-1] if long else point.price > closes[i-1]):
                    continue
                consumed = (lows[point.index+1:i] <= point.price).any() if long else (highs[point.index+1:i] >= point.price).any()
                if not consumed:
                    candidates.append(point)
            if not candidates:
                continue
            level = max(candidates,key=lambda point:point.price) if long else min(candidates,key=lambda point:point.price)
            reclaimed = lows[i] < level.price < closes[i] if long else highs[i] > level.price > closes[i]
            if reclaimed:
                day_events.append((direction,level,float(lows[i] if long else highs[i])))
        if day_events:
            events.append((i,day_events))
    if not events:
        return _Leg("no_actual_daily_sweep_reclaim")
    i,day_events = events[-1]
    index = pd.DatetimeIndex(indices,tz="UTC")
    origin = index[i]
    if len(day_events) != 1:
        return _Leg("ambiguous_double_daily_raid",origin_index=i,
            witness=(("context_origin_known_close_utc",(origin+pd.Timedelta(days=1)).isoformat()),))
    direction,level,extreme = day_events[0]
    long = direction == "LONG"
    base = dict(direction=direction,origin_index=i,origin_level=level.price,origin_extreme=extreme)
    witness = (("origin_evidence_tf","1d"),
        ("context_origin_known_close_utc",(origin+pd.Timedelta(days=1)).isoformat()),
        ("context_origin_high",float(highs[i])),("context_origin_low",float(lows[i])),
        ("context_swept_level_open_utc",index[level.index].isoformat()),
        ("context_swept_level_known_close_utc",(index[level.index]+pd.Timedelta(days=lookback+1)).isoformat()),
        ("pre_A_trend_required",False),("origin_requires_right_pivot_confirmation",False))
    if i+1 >= len(prices):
        return _Leg("immediate_confirmation_not_closed",witness=witness,**base)
    confirmation = index[i+1]
    if confirmation != origin+pd.Timedelta(days=1):
        return _Leg("missing_immediate_confirmation_day",witness=witness,**base)
    witness += (("context_confirmation_open_utc",confirmation.isoformat()),
        ("context_confirmation_known_close_utc",(confirmation+pd.Timedelta(days=1)).isoformat()),
        ("context_confirmation_high",float(highs[i+1])),("context_confirmation_low",float(lows[i+1])),
        ("context_confirmation_close",float(closes[i+1])),
        ("context_confirmation_rule","immediate next D1 HH/HL and close above A high" if long
         else "immediate next D1 LL/LH and close below A low"))
    valid = (highs[i+1] > highs[i] and lows[i+1] > lows[i] and closes[i+1] > highs[i] if long
             else lows[i+1] < lows[i] and highs[i+1] < highs[i] and closes[i+1] < lows[i])
    if not valid:
        return _Leg("immediate_confirmation_not_directional",witness=witness,**base)
    breached = (lows[i+1:] < extreme).any() if long else (highs[i+1:] > extreme).any()
    if breached:
        return _Leg("origin_extreme_broken",witness=witness,**base)
    if frozen_v3._completed_leg(points,i,direction):
        return _Leg("completed_new_leg_after_origin",witness=witness,**base)
    context = pd.DataFrame(prices,columns=["open","high","low","close"],index=index)
    targets = frozen_v3._current_targets(context,points,direction)
    return _Leg("context_ready" if targets else "no_current_leg_target",targets=targets,witness=witness,**base)


def _post_sweep_break_leg(context,lookback):
    return _cached_post_sweep_break_leg(*v1._immutable_ohlc(context),lookback)


def diagnose_volium_v4_from_df(*,symbol,frames,settings=None,mode="intraday",now=None,
                              parameters:V4Parameters|None=None,enforce_session_filter=True,
                              _collect_candidates=False) -> V4Diagnostic:
    parameters = parameters or V4Parameters()
    features = {"version":"experimental_v4","parameters":asdict(parameters),"mode":mode}
    def reject(reason):
        return V4Diagnostic(None,reason,features)
    if mode != "intraday":
        raise ValueError("V4 post-sweep hypothesis is defined only for intraday")
    current = pd.Timestamp(now or datetime.now(timezone.utc))
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    context_tf,liquidity_tf,entry_tf = "1d","1h","5m"
    if not {context_tf,liquidity_tf,entry_tf}.issubset(frames):
        return reject("missing_frames")
    lookback = int(v1._setting(settings,"volium_swing_lookback",2))
    context_limit = int(v1._setting(settings,"volium_context_lookback",80))
    if lookback < 1 or context_limit < 2*lookback+5:
        raise ValueError("Invalid v4 structural interpretation")
    entry = v1._closed(frames[entry_tf],entry_tf,current)
    if len(entry) < parameters.reaction_atr_period+3:
        return reject("insufficient_reaction_history")
    confirm = entry.iloc[-1]
    candle_range = float(confirm.high-confirm.low)
    fraction = abs(float(confirm.close-confirm.open))/candle_range if candle_range > 0 else 0.0
    features["confirm_body_fraction"] = fraction
    if candle_range <= 0 or fraction < parameters.reaction_min_body_ratio:
        return reject("confirm_body_fraction")
    signal_close = entry.index[-1]+pd.Timedelta(seconds=v1._DURATIONS[entry_tf])
    if current-signal_close >= pd.Timedelta(seconds=2*v1._DURATIONS[entry_tf]):
        return reject("stale_signal")
    timed = enforce_session_filter and v1._setting(settings,"volium_session_enabled",True)
    if timed and (not v1.in_volium_session(signal_close,settings) or not v1.in_volium_session(current,settings)):
        return reject("outside_session")
    context_closed = v1._closed(frames[context_tf],context_tf,current)
    context = context_closed.tail(context_limit)
    points = v1._pivots(context,lookback)
    origin_time = None
    price = float(confirm.close)
    origin_liquidity = v1._closed(frames[liquidity_tf],liquidity_tf,current)
    leg = _post_sweep_break_leg(context,lookback)
    if leg.origin_index is not None:
        origin_time = context.index[leg.origin_index]
        features.update(context_origin_open_utc=origin_time.isoformat(),context_origin_level=leg.origin_level,
                        context_origin_extreme=leg.origin_extreme)
    if leg.witness:
        features.update(dict(leg.witness))
    if leg.reason != "context_ready":
        return reject(leg.reason)
    direction = leg.direction
    origin_end = origin_time+pd.Timedelta(seconds=v1._DURATIONS[context_tf])
    liquidity = origin_liquidity
    if liquidity.empty:
        return reject("empty_liquidity_frame")
    for frame in (entry,liquidity):
        after_origin = frame.loc[frame.index >= origin_end]
        if (after_origin.low.lt(leg.origin_extreme).any() if direction == "LONG" else after_origin.high.gt(leg.origin_extreme).any()):
            return reject("origin_extreme_broken_lower_tf")
    candidates = [point for point in leg.targets if point.price > price] if direction == "LONG" else [point for point in leg.targets if point.price < price]
    if not candidates:
        return reject("no_directional_current_leg_target")
    intact = []
    for point in candidates:
        target_life = context.index[point.index]+pd.Timedelta(seconds=v1._DURATIONS[context_tf])
        consumed = False
        for frame in (entry,liquidity):
            after_target = frame.loc[frame.index >= max(origin_end,target_life)]
            if (after_target.high.ge(point.price).any() if direction == "LONG" else after_target.low.le(point.price).any()):
                consumed = True
                break
        if not consumed:
            intact.append(point)
    features["targets_consumed_lower_tf"] = len(candidates)-len(intact)
    if not intact:
        return reject("context_target_consumed_lower_tf")
    candidates = intact
    destination = min(candidates,key=lambda point:point.price) if direction == "LONG" else max(candidates,key=lambda point:point.price)
    context_target = destination.price
    target_time = context.index[destination.index]
    target_life = target_time+pd.Timedelta(seconds=v1._DURATIONS[context_tf])
    features.update(direction=direction,context_target_open_utc=target_time.isoformat(),
                    context_target=context_target,context_target_known_close_utc=(target_time+pd.Timedelta(seconds=(lookback+1)*v1._DURATIONS[context_tf])).isoformat(),
                    context_target_after_origin=bool(target_time > origin_time),
                    context_method=parameters.context_mode,context_target_life_begin_utc=target_life.isoformat())
    age = min(int(v1._setting(settings,"volium_sweep_max_age_bars",12)),parameters.reaction_max_bars+1)
    if age < 1:
        raise ValueError("Sweep maximum age must be positive")
    failure = "no_valid_liquidity_sweep"
    deepest = 0
    collected = []
    processed_raids = set()
    for sweep_i in range(len(entry)-1,max(0,len(entry)-age-1),-1):
        sweep_time = entry.index[sweep_i]
        if timed and not v1.in_volium_session(sweep_time,settings):
            continue
        known = liquidity.iloc[:liquidity.index.searchsorted(sweep_time-pd.Timedelta(seconds=v1._DURATIONS[liquidity_tf]),side="right")].tail(context_limit)
        pivots = v1._pivots(known,lookback)
        prior_price = float(entry.close.iloc[sweep_i-1])
        opposite = "low" if direction == "LONG" else "high"
        levels = [point for point in v1._unswept(known,pivots,opposite)
                  if (point.price < prior_price if direction == "LONG" else point.price > prior_price)]
        if not levels:
            continue
        level = max(levels,key=lambda point:point.price) if direction == "LONG" else min(levels,key=lambda point:point.price)
        sweep = entry.iloc[sweep_i]
        if not (sweep.low < level.price if direction == "LONG" else sweep.high > level.price):
            continue
        raid = _current_raid(entry,known,pivots,level,direction)
        if raid is None:
            continue
        first_i,actual_i,known_close = raid
        raid_start = entry.index[first_i]
        raid_key = (known.index[level.index],raid_start)
        if raid_key in processed_raids:
            continue
        processed_raids.add(raid_key)
        sweep_time = entry.index[actual_i]
        sweep = entry.iloc[actual_i]
        if timed and (not v1.in_volium_session(raid_start,settings) or not v1.in_volium_session(sweep_time,settings)):
            continue
        if not _raid_reaction(entry,first_i,actual_i,level.price,direction,parameters.reaction_max_bars,parameters.reaction_min_body_ratio):
            if deepest < 1:
                failure,deepest = "nonlinear_or_unconfirmed_reaction",1
            continue
        strength = _reaction_features(entry,first_i,direction,parameters)
        if strength is None:
            if deepest < 2:
                failure,deepest = "insufficient_pre_sweep_atr",2
            continue
        features.update(strength,sweep_open_utc=sweep_time.isoformat(),liquidity_level=level.price,
                        liquidity_level_open_utc=known.index[level.index].isoformat(),
                        raid_start_open_utc=raid_start.isoformat(),liquidity_known_close_utc=known_close.isoformat(),
                        raid_start_to_extreme_bars=actual_i-first_i,raid_extreme_age_bars=len(entry)-1-actual_i)
        if strength["confirm_body_atr"] < parameters.reaction_min_atr:
            if deepest < 3:
                failure,deepest = "reaction_below_atr",3
            continue
        if strength["confirm_body_prior_body_ratio"] < parameters.reaction_min_prior_body_ratio:
            if deepest < 4:
                failure,deepest = "reaction_below_prior_body",4
            continue
        target_point = _correction_origin(known,pivots,direction,origin_time)
        target = None if target_point is None else target_point.price
        if target is None:
            failure,deepest = "no_current_correction_origin",5
            continue
        after = entry.iloc[first_i:]
        if (target <= price or after.high.ge(target).any() if direction == "LONG" else target >= price or after.low.le(target).any()):
            failure,deepest = "correction_target_consumed",6
            continue
        buffer = float(v1._setting(settings,"volium_stop_buffer_bps",2))
        if not isfinite(buffer) or not 0 <= buffer < 10000:
            raise ValueError("Invalid v4 stop buffer")
        stop = float(sweep.low)*(1-buffer/10000) if direction == "LONG" else float(sweep.high)*(1+buffer/10000)
        limit = v1.rr2_entry(stop,target)
        if not (stop < limit <= price < target if direction == "LONG" else target < price <= limit < stop):
            failure,deepest = "unavailable_2r_retest_geometry",7
            continue
        features.update(entry=limit,stop=stop,target=target,context_target=context_target,
            correction_origin_open_utc=None if target_point is None else known.index[target_point.index].isoformat(),
            confirmed_close_utc=signal_close.isoformat())
        identifier = sha256(f"volium-v4:{mode}:{symbol}:{context_tf}:{raid_start.isoformat()}".encode()).hexdigest()[:20]
        setup = TradeSetup(id=identifier,timestamp=signal_close.to_pydatetime(),symbol=symbol,direction=direction,
            setup_type=f"VOLIUM_{mode.upper()}",htf_bias="BULLISH" if direction == "LONG" else "BEARISH",
            entry=limit,stop_loss=stop,take_profits=[target],risk_reward=2,confidence="MEDIUM",
            confluences=["EXPERIMENTAL v4; numeric source interpretation",features["context_method"],
                f"{liquidity_tf} nearest opposite sweep",f"{entry_tf} closed V body clearance",
                f"confirm body / pre-sweep ATR {strength['confirm_body_atr']:.4g}",
                f"confirm body / opposite body {strength['confirm_body_prior_body_ratio']:.4g}",
                f"Structural target mode {parameters.correction_target_mode}","Fixed sweep stop, structural target, 2R retest limit"])
        accepted = V4Diagnostic(setup,"accepted",features.copy())
        if _collect_candidates:
            collected.append(accepted)
        else:
            return accepted
    if _collect_candidates:
        features["structural_candidates"] = collected
        return V4Diagnostic(None,"candidate_features" if collected else failure,features)
    return reject(failure)


def analyze_volium_v4_from_df(**kwargs):
    return diagnose_volium_v4_from_df(**kwargs).setup


def make_v4_analyzer(parameters:V4Parameters|None=None):
    parameters = parameters or V4Parameters()
    def analyze(**kwargs):
        return analyze_volium_v4_from_df(**kwargs,parameters=parameters)
    return analyze


def diagnose_volium_v4_batch_from_df(*,parameter_sets,**kwargs):
    """Extract ALL broad structural candidates once per structural parameter group.

    A strict filter may reject the newest sweep but accept an older one. The
    batch never uses a single broad accepted setup as a rejection shortcut.
    ATR/anchor data precede the sweep; no outcomes enter candidate selection.
    """
    parameter_sets = list(parameter_sets)
    groups = {}
    for i,parameters in enumerate(parameter_sets):
        key = (parameters.context_mode,parameters.correction_target_mode,parameters.reaction_atr_period)
        groups.setdefault(key,[]).append((i,parameters))
    results = [None]*len(parameter_sets)
    for (context_mode,target_mode,atr_period),members in groups.items():
        broad = V4Parameters(context_mode=context_mode,correction_target_mode=target_mode,
            reaction_atr_period=atr_period,reaction_min_atr=0,reaction_min_prior_body_ratio=0,
            reaction_min_body_ratio=min(parameters.reaction_min_body_ratio for _,parameters in members),
            reaction_max_bars=max(parameters.reaction_max_bars for _,parameters in members))
        extracted = diagnose_volium_v4_from_df(**kwargs,parameters=broad,_collect_candidates=True)
        candidates = extracted.features.get("structural_candidates",[])
        for i,parameters in members:
            selected = None
            reason = extracted.reason if not candidates else "no_candidate_passed_v4_strength"
            rejected_features = {key:value for key,value in extracted.features.items() if key != "structural_candidates"}
            if extracted.features.get("confirm_body_fraction",1) < parameters.reaction_min_body_ratio:
                results[i] = V4Diagnostic(None,"confirm_body_fraction",{**rejected_features,"parameters":asdict(parameters)})
                continue
            for candidate in candidates:
                candidate_features = candidate.features
                if candidate_features["reaction_age_bars"] > parameters.reaction_max_bars or candidate_features["opposite_anchor_offset_bars"] > parameters.reaction_max_bars:
                    reason = "nonlinear_or_unconfirmed_reaction"
                elif candidate_features["confirm_body_fraction"] < parameters.reaction_min_body_ratio:
                    reason = "confirm_body_fraction"
                elif candidate_features["confirm_body_atr"] < parameters.reaction_min_atr:
                    reason = "reaction_below_atr"
                elif candidate_features["confirm_body_prior_body_ratio"] < parameters.reaction_min_prior_body_ratio:
                    reason = "reaction_below_prior_body"
                else:
                    selected = V4Diagnostic(candidate.setup.model_copy(deep=True),"accepted",
                        {**candidate_features,"parameters":asdict(parameters)})
                    break
                rejected_features = candidate_features
            if selected is None:
                selected = V4Diagnostic(None,reason,{**rejected_features,"parameters":asdict(parameters)})
            results[i] = selected
    return results


def analyze_volium_v4_batch_from_df(*,parameter_sets,**kwargs):
    return [result.setup for result in diagnose_volium_v4_batch_from_df(parameter_sets=parameter_sets,**kwargs)]
