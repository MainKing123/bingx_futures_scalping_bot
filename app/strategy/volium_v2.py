"""Experimental source-guided structural/strength diagnostics; v1 is unchanged.

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


@dataclass(frozen=True)
class V2Parameters:
    context_mode: Literal["preexisting_target", "legacy_context"] = "preexisting_target"
    correction_target_mode: Literal["leg_origin", "last_pivot"] = "leg_origin"
    reaction_min_atr: float = .8
    reaction_min_prior_body_ratio: float = 1.0
    reaction_atr_period: int = 14
    reaction_min_body_ratio: float = .6
    reaction_max_bars: int = 3

    def __post_init__(self):
        if self.context_mode not in {"preexisting_target","legacy_context"}:
            raise ValueError("Unsupported v2 context hypothesis")
        if self.correction_target_mode not in {"leg_origin","last_pivot"}:
            raise ValueError("Unsupported v2 correction hypothesis")
        if not all(isfinite(value) for value in (self.reaction_min_atr,self.reaction_min_prior_body_ratio,self.reaction_min_body_ratio)):
            raise ValueError("V2 thresholds must be finite")
        if not 0 <= self.reaction_min_atr <= 10 or not 0 <= self.reaction_min_prior_body_ratio <= 10:
            raise ValueError("V2 strength ratios must be between zero and ten")
        if any(isinstance(value,bool) or not isinstance(value,int) for value in (self.reaction_atr_period,self.reaction_max_bars)):
            raise ValueError("V2 candle counts must be integers")
        if not 0 < self.reaction_min_body_ratio <= 1 or not 2 <= self.reaction_atr_period <= 100 or not 1 <= self.reaction_max_bars <= 20:
            raise ValueError("Invalid v2 candle interpretation parameters")


@dataclass
class V2Diagnostic:
    setup: TradeSetup | None
    reason: str
    features: dict

    def as_dict(self):
        return {"accepted":self.setup is not None,"reason":self.reason,"features":self.features,
                "setup":self.setup.model_dump(mode="json") if self.setup is not None else None}


@dataclass(frozen=True)
class _Leg:
    reason: str
    direction: str | None = None
    origin_index: int | None = None
    origin_level: float | None = None
    origin_extreme: float | None = None
    targets: tuple = ()


@lru_cache(maxsize=2048)
def _cached_preexisting_leg(indices: tuple, prices: tuple, lookback: int) -> _Leg:
    """Most recent sweep with a causal pre-A trend; no fallback to an older A."""
    if len(prices) < 2*lookback+5:
        return _Leg("insufficient_context")
    matrix = np.asarray(prices,dtype=float)
    highs,lows,closes = matrix[:,1],matrix[:,2],matrix[:,3]
    points = v1._cached_pivots(indices,prices,lookback)
    events = []
    for i in range(2*lookback+2,len(prices)):
        before = [point for point in points if point.index+lookback < i]
        direction = v1._trend(before)
        if direction is None:
            continue
        long = direction == "LONG"
        kind = "low" if long else "high"
        candidates = []
        for point in before:
            if point.kind != kind or not (point.price < closes[i-1] if long else point.price > closes[i-1]):
                continue
            touched = (lows[point.index+1:i] <= point.price).any() if long else (highs[point.index+1:i] >= point.price).any()
            if not touched:
                candidates.append(point)
        if not candidates:
            continue
        level = (max(candidates,key=lambda point:point.price) if long else min(candidates,key=lambda point:point.price))
        if (lows[i] < level.price < closes[i] if long else highs[i] > level.price > closes[i]):
            events.append((i,direction,level.price,float(lows[i] if long else highs[i]),before))
    if not events:
        return _Leg("no_causal_origin")
    i,direction,level,extreme,before = events[-1]
    long = direction == "LONG"
    base = dict(direction=direction,origin_index=i,origin_level=level,origin_extreme=extreme)
    if ((lows[i+1:] < extreme).any() if long else (highs[i+1:] > extreme).any()):
        return _Leg("origin_extreme_broken",**base)
    # One impulse and its current pullback may still be the same leg. Two
    # confirmed directional peaks separated by a new opposite pivot are a
    # completed intervening leg. This scale is an explicit fractal hypothesis.
    directional = "high" if long else "low"
    after = [point for point in points if point.index > i]
    for first in after:
        if first.kind != directional:
            continue
        opposite = next((point for point in after if point.index > first.index and point.kind != directional),None)
        if opposite is not None and any(point.index > opposite.index and point.kind == directional for point in after):
            return _Leg("completed_new_leg_after_origin",**base)
    targets = []
    for point in before:
        if point.kind != directional:
            continue
        touched = (highs[point.index+1:] >= point.price).any() if long else (lows[point.index+1:] <= point.price).any()
        if not touched:
            targets.append(point)
    if not targets:
        return _Leg("preexisting_target_consumed",**base)
    return _Leg("context_ready",targets=tuple(targets),**base)


def _preexisting_leg(df,lookback):
    return _cached_preexisting_leg(*v1._immutable_ohlc(df),lookback)


def _correction_origin(df,points,direction,earliest=None):
    """Latest advancing structural peak before the current corrective sequence.

    Lower highs (or higher lows for a short) after that peak belong to the
    correction. A first peak is a fallback when no prior same-kind pivot exists.
    """
    long = direction == "LONG"
    directional = [point for point in points if point.kind == ("high" if long else "low")]
    if earliest is not None:
        directional = [point for point in directional if df.index[point.index] >= earliest]
    if not directional:
        return None
    origins = [directional[0]]
    for previous,current in zip(directional,directional[1:]):
        if current.price > previous.price if long else current.price < previous.price:
            origins.append(current)
    origin = origins[-1]
    later = df.high.iloc[origin.index+1:] if long else df.low.iloc[origin.index+1:]
    if (later.ge(origin.price).any() if long else later.le(origin.price).any()):
        return None
    return origin


def _reaction_features(df,sweep_i,direction,parameters):
    start = sweep_i-parameters.reaction_atr_period
    if start < 1:
        return None
    matrix = df[["open","high","low","close"]].to_numpy(dtype=float)
    before = matrix[start:sweep_i]
    previous_closes = matrix[start-1:sweep_i-1,3]
    tr = np.maximum(before[:,1]-before[:,2],np.maximum(np.abs(before[:,1]-previous_closes),np.abs(before[:,2]-previous_closes)))
    atr = float(tr.mean())
    anchor = None
    for i in range(sweep_i,max(-1,sweep_i-parameters.reaction_max_bars-1),-1):
        opposite = matrix[i,3] < matrix[i,0] if direction == "LONG" else matrix[i,3] > matrix[i,0]
        if opposite and i < len(matrix)-1:
            anchor = i
            break
    if anchor is None or atr <= 0:
        return None
    body = abs(float(matrix[-1,3]-matrix[-1,0]))
    prior_body = abs(float(matrix[anchor,3]-matrix[anchor,0]))
    return {"pre_sweep_atr":atr,"confirm_body":body,"confirm_body_atr":body/atr,
            "opposite_body":prior_body,"confirm_body_prior_body_ratio":body/prior_body,
            "reaction_age_bars":len(matrix)-1-sweep_i,"opposite_anchor_offset_bars":sweep_i-anchor,
            "confirm_body_fraction":body/float(matrix[-1,1]-matrix[-1,2]),
            "atr_last_open_utc":df.index[sweep_i-1].isoformat(),"opposite_body_open_utc":df.index[anchor].isoformat()}


def diagnose_volium_v2_from_df(*,symbol,frames,settings=None,mode="intraday",now=None,
                              parameters:V2Parameters|None=None,enforce_session_filter=True,
                              _collect_candidates=False) -> V2Diagnostic:
    parameters = parameters or V2Parameters()
    features = {"version":"experimental_v2","parameters":asdict(parameters),"mode":mode}
    def reject(reason):
        return V2Diagnostic(None,reason,features)
    if mode not in {"intraday","scalp","swing"}:
        raise ValueError("Unsupported VOLIUM mode")
    current = pd.Timestamp(now or datetime.now(timezone.utc))
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    weekly = mode == "swing" and v1._setting(settings,"volium_swing_context","1d") == "1w"
    context_tf,liquidity_tf,entry_tf = (("1d","1h","5m") if mode == "intraday" else
        ("1h","5m","1m") if mode == "scalp" else ("1w","1w","4h") if weekly else ("1d","1d","1h"))
    if not {context_tf,liquidity_tf,entry_tf}.issubset(frames):
        return reject("missing_frames")
    lookback = int(v1._setting(settings,"volium_swing_lookback",2))
    context_limit = int(v1._setting(settings,"volium_context_lookback",80))
    if lookback < 1 or context_limit < 2*lookback+5:
        raise ValueError("Invalid v2 structural interpretation")
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
    timed = mode != "swing" and enforce_session_filter and v1._setting(settings,"volium_session_enabled",True)
    if timed and (not v1.in_volium_session(signal_close,settings) or not v1.in_volium_session(current,settings)):
        return reject("outside_session")
    context_closed = v1._closed(frames[context_tf],context_tf,current)
    context = context_closed.tail(context_limit)
    points = v1._pivots(context,lookback)
    origin_time = None
    price = float(confirm.close)
    if mode != "scalp" and parameters.context_mode == "preexisting_target":
        leg = _preexisting_leg(context,lookback)
        if leg.origin_index is not None:
            origin_time = context.index[leg.origin_index]
            features.update(context_origin_open_utc=origin_time.isoformat(),context_origin_level=leg.origin_level,
                            context_origin_extreme=leg.origin_extreme)
        if leg.reason != "context_ready":
            return reject(leg.reason)
        direction = leg.direction
        candidates = [point for point in leg.targets if point.price > price] if direction == "LONG" else [point for point in leg.targets if point.price < price]
        if not candidates:
            return reject("no_directional_preexisting_target")
        destination = min(candidates,key=lambda point:point.price) if direction == "LONG" else max(candidates,key=lambda point:point.price)
        context_target = destination.price
        features.update(context_target_open_utc=context.index[destination.index].isoformat(),context_target=context_target,
                        context_method="pre-A trend and preexisting unswept target")
    else:
        direction = v1._trend(points)
        if direction is None:
            return reject("no_confirmed_context_trend")
        context_target = v1._target(context,points,direction,price) if mode != "scalp" else None
        if mode != "scalp" and context_target is None:
            return reject("no_context_target")
        if mode == "intraday" and v1._setting(settings,"volium_require_daily_origin_sweep",True) and not v1._origin_sweep(context,direction,lookback):
            return reject("no_legacy_origin")
        features["context_method"] = "H1 trend and active progress" if mode == "scalp" else "v1 legacy context ablation"
    features["direction"] = direction
    if mode == "scalp" and not v1._active_trend(context,direction,float(v1._setting(settings,"volium_active_trend_min_atr",1))):
        return reject("inactive_hourly_trend")
    liquidity = context_closed if liquidity_tf == context_tf else v1._closed(frames[liquidity_tf],liquidity_tf,current)
    if liquidity.empty:
        return reject("empty_liquidity_frame")
    if origin_time is not None:
        # Closed lower-TF bars cover the forming daily/weekly candle too; do not
        # treat its B as unhit merely because that HTF candle is still forming.
        for frame in (entry,liquidity):
            after = frame.loc[frame.index >= origin_time]
            if (after.high.ge(context_target).any() if direction == "LONG" else after.low.le(context_target).any()):
                return reject("context_target_consumed_lower_tf")
    age = min(int(v1._setting(settings,"volium_sweep_max_age_bars",12)),parameters.reaction_max_bars+1)
    if age < 1:
        raise ValueError("Sweep maximum age must be positive")
    failure = "no_valid_liquidity_sweep"
    deepest = 0
    collected = []
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
        if not v1._reaction(entry,sweep_i,level.price,direction,parameters.reaction_max_bars,parameters.reaction_min_body_ratio):
            if deepest < 1:
                failure,deepest = "nonlinear_or_unconfirmed_reaction",1
            continue
        strength = _reaction_features(entry,sweep_i,direction,parameters)
        if strength is None:
            if deepest < 2:
                failure,deepest = "insufficient_pre_sweep_atr",2
            continue
        features.update(strength,sweep_open_utc=sweep_time.isoformat(),liquidity_level=level.price,
                        liquidity_level_open_utc=known.index[level.index].isoformat())
        if strength["confirm_body_atr"] < parameters.reaction_min_atr:
            if deepest < 3:
                failure,deepest = "reaction_below_atr",3
            continue
        if strength["confirm_body_prior_body_ratio"] < parameters.reaction_min_prior_body_ratio:
            if deepest < 4:
                failure,deepest = "reaction_below_prior_body",4
            continue
        target_point = None
        if mode == "scalp":
            target = v1._target(known,pivots,direction,prior_price)
            context_target = target
        elif parameters.correction_target_mode == "last_pivot":
            candidates = [point for point in pivots if point.kind == ("high" if direction == "LONG" else "low")]
            target_point = candidates[-1] if candidates else None
            target = None if target_point is None else target_point.price
        else:
            target_point = _correction_origin(known,pivots,direction,origin_time)
            target = None if target_point is None else target_point.price
        if target is None:
            failure,deepest = "no_current_correction_origin",5
            continue
        after = entry.iloc[sweep_i:]
        if (target <= price or after.high.ge(target).any() if direction == "LONG" else target >= price or after.low.le(target).any()):
            failure,deepest = "correction_target_consumed",6
            continue
        buffer = float(v1._setting(settings,"volium_scalp_stop_buffer_bps" if mode == "scalp" else "volium_stop_buffer_bps",10 if mode == "scalp" else 2))
        if not isfinite(buffer) or not 0 <= buffer < 10000:
            raise ValueError("Invalid v2 stop buffer")
        stop = float(sweep.low)*(1-buffer/10000) if direction == "LONG" else float(sweep.high)*(1+buffer/10000)
        limit = v1.rr2_entry(stop,target)
        if not (stop < limit <= price < target if direction == "LONG" else target < price <= limit < stop):
            failure,deepest = "unavailable_2r_retest_geometry",7
            continue
        features.update(entry=limit,stop=stop,target=target,context_target=context_target,
            correction_origin_open_utc=None if target_point is None else known.index[target_point.index].isoformat(),
            confirmed_close_utc=signal_close.isoformat())
        identifier = sha256(f"volium-v2:{mode}:{symbol}:{context_tf}:{sweep_time.isoformat()}".encode()).hexdigest()[:20]
        setup = TradeSetup(id=identifier,timestamp=signal_close.to_pydatetime(),symbol=symbol,direction=direction,
            setup_type=f"VOLIUM_{mode.upper()}",htf_bias="BULLISH" if direction == "LONG" else "BEARISH",
            entry=limit,stop_loss=stop,take_profits=[target],risk_reward=2,confidence="MEDIUM",
            confluences=["EXPERIMENTAL v2; numeric source interpretation",features["context_method"],
                f"{liquidity_tf} nearest opposite sweep",f"{entry_tf} closed V body clearance",
                f"confirm body / pre-sweep ATR {strength['confirm_body_atr']:.4g}",
                f"confirm body / opposite body {strength['confirm_body_prior_body_ratio']:.4g}",
                f"Structural target mode {parameters.correction_target_mode}","Fixed sweep stop, structural target, 2R retest limit"])
        accepted = V2Diagnostic(setup,"accepted",features.copy())
        if _collect_candidates:
            collected.append(accepted)
        else:
            return accepted
    if _collect_candidates:
        features["structural_candidates"] = collected
        return V2Diagnostic(None,"candidate_features" if collected else failure,features)
    return reject(failure)


def analyze_volium_v2_from_df(**kwargs):
    return diagnose_volium_v2_from_df(**kwargs).setup


def make_v2_analyzer(parameters:V2Parameters|None=None):
    parameters = parameters or V2Parameters()
    def analyze(**kwargs):
        return analyze_volium_v2_from_df(**kwargs,parameters=parameters)
    return analyze


def diagnose_volium_v2_batch_from_df(*,parameter_sets,**kwargs):
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
        broad = V2Parameters(context_mode=context_mode,correction_target_mode=target_mode,
            reaction_atr_period=atr_period,reaction_min_atr=0,reaction_min_prior_body_ratio=0,
            reaction_min_body_ratio=min(parameters.reaction_min_body_ratio for _,parameters in members),
            reaction_max_bars=max(parameters.reaction_max_bars for _,parameters in members))
        extracted = diagnose_volium_v2_from_df(**kwargs,parameters=broad,_collect_candidates=True)
        candidates = extracted.features.get("structural_candidates",[])
        for i,parameters in members:
            selected = None
            reason = extracted.reason if not candidates else "no_candidate_passed_v2_strength"
            rejected_features = {key:value for key,value in extracted.features.items() if key != "structural_candidates"}
            if extracted.features.get("confirm_body_fraction",1) < parameters.reaction_min_body_ratio:
                results[i] = V2Diagnostic(None,"confirm_body_fraction",{**rejected_features,"parameters":asdict(parameters)})
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
                    selected = V2Diagnostic(candidate.setup.model_copy(deep=True),"accepted",
                        {**candidate_features,"parameters":asdict(parameters)})
                    break
                rejected_features = candidate_features
            if selected is None:
                selected = V2Diagnostic(None,reason,{**rejected_features,"parameters":asdict(parameters)})
            results[i] = selected
    return results


def analyze_volium_v2_batch_from_df(*,parameter_sets,**kwargs):
    return [result.setup for result in diagnose_volium_v2_batch_from_df(parameter_sets=parameter_sets,**kwargs)]
