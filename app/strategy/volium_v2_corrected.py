"""Fixed-default research control: frozen v2 plus its lower-TF A invariant.

This module never changes the original v2. It adds no strategy hypothesis or
strength grid and is ineligible for follow-up candidate selection.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.strategy import volium as v1
from app.strategy import volium_v2 as frozen_v2

V2Parameters = frozen_v2.V2Parameters
V2Diagnostic = frozen_v2.V2Diagnostic


def _guard(result,kwargs):
    if result.setup is None:
        return result
    features = dict(result.features)
    features["version"] = "corrected_v2_control"
    features["lower_tf_origin_guard"] = True
    origin = features.get("context_origin_open_utc")
    extreme = features.get("context_origin_extreme")
    if origin is None or extreme is None:
        return V2Diagnostic(result.setup,result.reason,features)
    current = pd.Timestamp(kwargs.get("now") or datetime.now(timezone.utc))
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    mode = kwargs.get("mode","intraday")
    settings = kwargs.get("settings")
    weekly = mode == "swing" and v1._setting(settings,"volium_swing_context","1d") == "1w"
    context_tf = "1w" if weekly else "1d"
    origin_end = pd.Timestamp(origin)+pd.Timedelta(seconds=v1._DURATIONS[context_tf])
    lower_tfs = ("1h","5m") if mode == "intraday" else ("4h",) if weekly else ("1h",)
    for timeframe in lower_tfs:
        if timeframe not in kwargs["frames"]:
            continue
        closed = v1._closed(kwargs["frames"][timeframe],timeframe,current)
        after = closed.loc[closed.index >= origin_end]
        breached = after.low.lt(extreme).any() if result.setup.direction == "LONG" else after.high.gt(extreme).any()
        if breached:
            features["origin_breach_tf"] = timeframe
            return V2Diagnostic(None,"origin_extreme_broken_lower_tf",features)
    return V2Diagnostic(result.setup,result.reason,features)


def diagnose_volium_v2_corrected_from_df(**kwargs):
    parameters = kwargs.get("parameters") or V2Parameters()
    if parameters.context_mode != "preexisting_target":
        raise ValueError("Corrected v2 control requires its original linked context")
    return _guard(frozen_v2.diagnose_volium_v2_from_df(**kwargs),kwargs)


def analyze_volium_v2_corrected_from_df(**kwargs):
    return diagnose_volium_v2_corrected_from_df(**kwargs).setup


def make_v2_corrected_analyzer(parameters=None):
    parameters = parameters or V2Parameters()
    def analyze(**kwargs):
        return analyze_volium_v2_corrected_from_df(**kwargs,parameters=parameters)
    return analyze


def diagnose_volium_v2_corrected_batch_from_df(*,parameter_sets,**kwargs):
    parameter_sets = list(parameter_sets)
    if any(parameters.context_mode != "preexisting_target" for parameters in parameter_sets):
        raise ValueError("Corrected v2 control requires its original linked context")
    return [_guard(result,kwargs) for result in frozen_v2.diagnose_volium_v2_batch_from_df(parameter_sets=parameter_sets,**kwargs)]


def analyze_volium_v2_corrected_batch_from_df(*,parameter_sets,**kwargs):
    return [result.setup for result in diagnose_volium_v2_corrected_batch_from_df(parameter_sets=parameter_sets,**kwargs)]
