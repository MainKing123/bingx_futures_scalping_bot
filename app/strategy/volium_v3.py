"""Experimental current-leg hypotheses; frozen v1/v2 are unchanged.

Numeric thresholds and the definition of a completed leg are hypotheses, not
settings disclosed by the video author. This module has no execution or IO.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from math import isfinite
from typing import Literal

import numpy as np
import pandas as pd

from app.schemas.setup import TradeSetup
from app.strategy import volium as v1
from app.strategy import volium_v2 as frozen_v2


@dataclass(frozen=True)
class V3Parameters:
    context_mode: Literal["latest_leg_daily", "latest_leg_local"] = "latest_leg_daily"
    correction_target_mode: Literal["leg_origin"] = "leg_origin"
    reaction_min_atr: float = .8
    reaction_min_prior_body_ratio: float = 1.0
    reaction_atr_period: int = 14
    reaction_min_body_ratio: float = .6
    reaction_max_bars: int = 3

    def __post_init__(self):
        if self.context_mode not in {"latest_leg_daily","latest_leg_local"}:
            raise ValueError("Unsupported v3 context hypothesis")
        if self.correction_target_mode not in {"leg_origin"}:
            raise ValueError("Unsupported v3 correction hypothesis")
        if not all(isfinite(value) for value in (self.reaction_min_atr,self.reaction_min_prior_body_ratio,self.reaction_min_body_ratio)):
            raise ValueError("V3 thresholds must be finite")
        if not 0 <= self.reaction_min_atr <= 10 or not 0 <= self.reaction_min_prior_body_ratio <= 10:
            raise ValueError("V3 strength ratios must be between zero and ten")
        if any(isinstance(value,bool) or not isinstance(value,int) for value in (self.reaction_atr_period,self.reaction_max_bars)):
            raise ValueError("V3 candle counts must be integers")
        if not 0 < self.reaction_min_body_ratio <= 1 or not 2 <= self.reaction_atr_period <= 100 or not 1 <= self.reaction_max_bars <= 20:
            raise ValueError("Invalid v3 candle interpretation parameters")


@dataclass
class V3Diagnostic:
    setup: TradeSetup | None
    reason: str
    features: dict

    def as_dict(self):
        return {"accepted":self.setup is not None,"reason":self.reason,"features":self.features,
                "setup":self.setup.model_dump(mode="json") if self.setup is not None else None}


# Immutable structural extraction is implemented below. Relative V strength
# and correction scale reuse the unchanged causal helpers from frozen v2.
_reaction_features = frozen_v2._reaction_features
_correction_origin = frozen_v2._correction_origin


@dataclass(frozen=True)
class _Leg:
    reason: str
    direction: str | None = None
    origin_index: int | None = None
    origin_level: float | None = None
    origin_extreme: float | None = None
    targets: tuple = ()
    witness: tuple = ()


def _current_targets(context,points,direction):
    return tuple(v1._unswept(context,points,"high" if direction == "LONG" else "low"))


def _completed_leg(points,origin_index,direction):
    directional = "high" if direction == "LONG" else "low"
    after = [point for point in points if point.index > origin_index]
    for first in after:
        if first.kind != directional:
            continue
        opposite = next((point for point in after if point.index > first.index and point.kind != directional),None)
        if opposite is not None and any(point.index > opposite.index and point.kind == directional for point in after):
            return True
    return False


@lru_cache(maxsize=2048)
def _cached_local_witness(indices,prices,origin_iso,extreme,direction,lookback,context_limit):
    """Evidence from only the origin day and its preceding H1 history."""
    frame = pd.DataFrame(prices,columns=["open","high","low","close"],index=pd.DatetimeIndex(indices,tz="UTC"))
    origin = pd.Timestamp(origin_iso)
    long = direction == "LONG"
    events = []
    for i in range(len(frame)):
        stamp = frame.index[i]
        if not origin <= stamp < origin+pd.Timedelta(days=1):
            continue
        known = frame.iloc[:i].tail(context_limit)
        if len(known) < context_limit:
            continue
        points = v1._pivots(known,lookback)
        previous = float(frame.close.iloc[i-1])
        candidates = [point for point in v1._unswept(known,points,"low" if long else "high")
                      if (point.price < previous if long else point.price > previous)]
        if not candidates:
            continue
        level = max(candidates,key=lambda point:point.price) if long else min(candidates,key=lambda point:point.price)
        candle = frame.iloc[i]
        swept = candle.low < level.price < candle.close if long else candle.high > level.price > candle.close
        matches_extreme = abs(float(candle.low if long else candle.high)-extreme) <= max(abs(extreme),1)*1e-10
        if swept and matches_extreme:
            events.append((level.price,stamp.isoformat(),known.index[level.index].isoformat()))
    return events[-1] if events else None


def _latest_leg(context,hourly,lookback,context_limit,context_mode):
    points = v1._pivots(context,lookback)
    if context_mode == "latest_leg_daily":
        original = frozen_v2._preexisting_leg(context,lookback)
        base = dict(direction=original.direction,origin_index=original.origin_index,
                    origin_level=original.origin_level,origin_extreme=original.origin_extreme)
        if original.reason not in {"context_ready","preexisting_target_consumed"}:
            return _Leg(original.reason,**base)
        targets = _current_targets(context,points,original.direction)
        return _Leg("context_ready" if targets else "no_current_leg_target",targets=targets,
                    witness=(("origin_evidence_tf","1d"),
                             ("context_origin_known_close_utc",(context.index[original.origin_index]+pd.Timedelta(days=1)).isoformat())),**base)
    events = []
    for point in points:
        before = [prior for prior in points if prior.index+lookback < point.index]
        direction = v1._trend(before)
        if direction is not None and point.kind == ("low" if direction == "LONG" else "high"):
            events.append((point,before,direction))
    if not events:
        return _Leg("no_confirmed_local_origin")
    point,before,direction = events[-1]
    long = direction == "LONG"
    base = dict(direction=direction,origin_index=point.index,origin_extreme=point.price)
    previous = [prior for prior in before if prior.kind == point.kind]
    if not previous or not (point.price > previous[-1].price if long else point.price < previous[-1].price):
        return _Leg("local_origin_not_hl_lh",**base)
    after_context = context.iloc[point.index+1:]
    if (after_context.low.lt(point.price).any() if long else after_context.high.gt(point.price).any()):
        return _Leg("origin_extreme_broken",**base)
    if _completed_leg(points,point.index,direction):
        return _Leg("completed_new_leg_after_origin",**base)
    origin = context.index[point.index]
    origin_end = origin+pd.Timedelta(days=1)
    prior = hourly.iloc[:hourly.index.searchsorted(origin-pd.Timedelta(hours=1),side="right")]
    if len(prior) < context_limit:
        return _Leg("insufficient_origin_liquidity_history",**base)
    witness_frame = hourly.iloc[:hourly.index.searchsorted(origin_end-pd.Timedelta(hours=1),side="right")].tail(context_limit+24)
    witness = _cached_local_witness(*v1._immutable_ohlc(witness_frame),origin.isoformat(),point.price,direction,lookback,context_limit)
    if witness is None:
        return _Leg("local_origin_without_actual_sweep",**base)
    level,sweep_time,level_time = witness
    targets = _current_targets(context,points,direction)
    return _Leg("context_ready" if targets else "no_current_leg_target",origin_level=level,targets=targets,
                witness=(("origin_evidence_tf","1h"),("origin_sweep_open_utc",sweep_time),
                         ("origin_swept_level_open_utc",level_time),
                         ("context_origin_known_close_utc",(origin+pd.Timedelta(days=lookback+1)).isoformat())),**base)


def _current_raid(entry,known,pivots,level,direction):
    """A known H1 level is consumed by its first M5 breach, not each repeat.

    Group only the closed prefix after the last known H1 close. This also
    captures its still-forming H1 candle without inventing additional raids.
    """
    known_close = known.index[-1]+pd.Timedelta(hours=1)
    if entry.index[0] > known_close:
        return None
    start = int(entry.index.searchsorted(known_close,side="left"))
    matrix = entry[["open","high","low","close"]].to_numpy(dtype=float)
    long = direction == "LONG"
    breached = matrix[start:,2] < level.price if long else matrix[start:,1] > level.price
    hits = np.flatnonzero(breached)
    if not len(hits):
        return None
    first = start+int(hits[0])
    if first < 1:
        return None
    prior_price = matrix[first-1,3]
    eligible = [point for point in v1._unswept(known,pivots,"low" if long else "high")
                if (point.price < prior_price if long else point.price > prior_price)]
    if not eligible:
        return None
    nearest = max(eligible,key=lambda point:point.price) if long else min(eligible,key=lambda point:point.price)
    if nearest != level:
        return None
    actual = first+int(np.argmin(matrix[first:,2]) if long else np.argmax(matrix[first:,1]))
    return first,actual,known_close


def _raid_reaction(entry,first,actual,level,direction,max_bars,min_body):
    """Bound total wait from the initial breach; clear its original body.

    The descending/ascending raid may deepen before the direct V recovery.
    Recovery closes must be strictly monotone from the actual known extreme.
    """
    last = len(entry)-1
    if last-first > max_bars:
        return False
    confirm = entry.iloc[-1]
    candle_range = float(confirm.high-confirm.low)
    if candle_range <= 0 or abs(float(confirm.close-confirm.open))/candle_range < min_body:
        return False
    anchor = None
    for i in range(first,max(-1,first-max_bars-1),-1):
        row = entry.iloc[i]
        opposite = row.close < row.open if direction == "LONG" else row.close > row.open
        if opposite and i < last:
            anchor = row
            break
    if anchor is None:
        return False
    recovery = entry.iloc[actual:]
    whole = entry.iloc[first:]
    changes = recovery.close.diff().dropna()
    if direction == "LONG":
        return bool(confirm.close > confirm.open and confirm.close > level and confirm.close >= anchor.open
                    and whole.open.min() <= anchor.close and (changes > 0).all()
                    and recovery.low.iloc[1:].ge(entry.low.iloc[actual]).all())
    return bool(confirm.close < confirm.open and confirm.close < level and confirm.close <= anchor.open
                and whole.open.max() >= anchor.close and (changes < 0).all()
                and recovery.high.iloc[1:].le(entry.high.iloc[actual]).all())


def diagnose_volium_v3_from_df(*,symbol,frames,settings=None,mode="intraday",now=None,
                              parameters:V3Parameters|None=None,enforce_session_filter=True,
                              _collect_candidates=False) -> V3Diagnostic:
    parameters = parameters or V3Parameters()
    features = {"version":"experimental_v3","parameters":asdict(parameters),"mode":mode}
    def reject(reason):
        return V3Diagnostic(None,reason,features)
    if mode != "intraday":
        raise ValueError("V3 current-leg hypotheses are defined only for intraday")
    current = pd.Timestamp(now or datetime.now(timezone.utc))
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    context_tf,liquidity_tf,entry_tf = "1d","1h","5m"
    if not {context_tf,liquidity_tf,entry_tf}.issubset(frames):
        return reject("missing_frames")
    lookback = int(v1._setting(settings,"volium_swing_lookback",2))
    context_limit = int(v1._setting(settings,"volium_context_lookback",80))
    if lookback < 1 or context_limit < 2*lookback+5:
        raise ValueError("Invalid v3 structural interpretation")
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
    leg = _latest_leg(context,origin_liquidity,lookback,context_limit,parameters.context_mode)
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
            raise ValueError("Invalid v3 stop buffer")
        stop = float(sweep.low)*(1-buffer/10000) if direction == "LONG" else float(sweep.high)*(1+buffer/10000)
        limit = v1.rr2_entry(stop,target)
        if not (stop < limit <= price < target if direction == "LONG" else target < price <= limit < stop):
            failure,deepest = "unavailable_2r_retest_geometry",7
            continue
        features.update(entry=limit,stop=stop,target=target,context_target=context_target,
            correction_origin_open_utc=None if target_point is None else known.index[target_point.index].isoformat(),
            confirmed_close_utc=signal_close.isoformat())
        identifier = sha256(f"volium-v3:{mode}:{symbol}:{context_tf}:{raid_start.isoformat()}".encode()).hexdigest()[:20]
        setup = TradeSetup(id=identifier,timestamp=signal_close.to_pydatetime(),symbol=symbol,direction=direction,
            setup_type=f"VOLIUM_{mode.upper()}",htf_bias="BULLISH" if direction == "LONG" else "BEARISH",
            entry=limit,stop_loss=stop,take_profits=[target],risk_reward=2,confidence="MEDIUM",
            confluences=["EXPERIMENTAL v3; numeric source interpretation",features["context_method"],
                f"{liquidity_tf} nearest opposite sweep",f"{entry_tf} closed V body clearance",
                f"confirm body / pre-sweep ATR {strength['confirm_body_atr']:.4g}",
                f"confirm body / opposite body {strength['confirm_body_prior_body_ratio']:.4g}",
                f"Structural target mode {parameters.correction_target_mode}","Fixed sweep stop, structural target, 2R retest limit"])
        accepted = V3Diagnostic(setup,"accepted",features.copy())
        if _collect_candidates:
            collected.append(accepted)
        else:
            return accepted
    if _collect_candidates:
        features["structural_candidates"] = collected
        return V3Diagnostic(None,"candidate_features" if collected else failure,features)
    return reject(failure)


def analyze_volium_v3_from_df(**kwargs):
    return diagnose_volium_v3_from_df(**kwargs).setup


def make_v3_analyzer(parameters:V3Parameters|None=None):
    parameters = parameters or V3Parameters()
    def analyze(**kwargs):
        return analyze_volium_v3_from_df(**kwargs,parameters=parameters)
    return analyze


def diagnose_volium_v3_batch_from_df(*,parameter_sets,**kwargs):
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
        broad = V3Parameters(context_mode=context_mode,correction_target_mode=target_mode,
            reaction_atr_period=atr_period,reaction_min_atr=0,reaction_min_prior_body_ratio=0,
            reaction_min_body_ratio=min(parameters.reaction_min_body_ratio for _,parameters in members),
            reaction_max_bars=max(parameters.reaction_max_bars for _,parameters in members))
        extracted = diagnose_volium_v3_from_df(**kwargs,parameters=broad,_collect_candidates=True)
        candidates = extracted.features.get("structural_candidates",[])
        for i,parameters in members:
            selected = None
            reason = extracted.reason if not candidates else "no_candidate_passed_v3_strength"
            rejected_features = {key:value for key,value in extracted.features.items() if key != "structural_candidates"}
            if extracted.features.get("confirm_body_fraction",1) < parameters.reaction_min_body_ratio:
                results[i] = V3Diagnostic(None,"confirm_body_fraction",{**rejected_features,"parameters":asdict(parameters)})
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
                    selected = V3Diagnostic(candidate.setup.model_copy(deep=True),"accepted",
                        {**candidate_features,"parameters":asdict(parameters)})
                    break
                rejected_features = candidate_features
            if selected is None:
                selected = V3Diagnostic(None,reason,{**rejected_features,"parameters":asdict(parameters)})
            results[i] = selected
    return results


def analyze_volium_v3_batch_from_df(*,parameter_sets,**kwargs):
    return [result.setup for result in diagnose_volium_v3_batch_from_df(parameter_sets=parameter_sets,**kwargs)]
