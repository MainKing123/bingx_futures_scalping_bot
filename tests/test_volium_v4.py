from dataclasses import replace
from datetime import datetime,timezone
from itertools import product

import pandas as pd
import pytest

from app.strategy import volium as v1
from app.strategy.volium_v4 import (V4Parameters,_post_sweep_break_leg,
    diagnose_volium_v4_from_df,analyze_volium_v4_from_df,analyze_volium_v4_batch_from_df,make_v4_analyzer)


NOW=datetime(2026,10,4,7,15,tzinfo=timezone.utc)


def _frame(highs,lows,end,freq):
    index=pd.date_range(end=end,periods=len(highs),freq=freq,tz="UTC")
    middle=[(high+low)/2 for high,low in zip(highs,lows)]
    return pd.DataFrame({"open":middle,"high":highs,"low":lows,"close":middle},index=index).astype(float)


def _frames():
    daily=_frame([110,111,140,113,114,120,116,115,125,120],
                 [101,100,90,100,101,99,100,89,95,94],"2026-10-04","D")
    daily.iloc[7]=[108,115,89,110]  # Latest actual LONG A, reclaimed old low90.
    daily.iloc[8]=[111,125,95,123]  # Immediate next day HH/HL and close>A.high.
    hourly=_frame([119,121,118,125,121,120,118,116,130,120,121,122,123],
                  [115,112,105,113,113,114,113,110,115,115,116,116,117],"2026-10-04 06:00","h")
    warmup=pd.DataFrame([(117.8,118.4,117.4,118.)]*16,columns=["open","high","low","close"],
                        index=pd.date_range("2026-10-04 05:35",periods=16,freq="5min",tz="UTC"))
    signal=pd.DataFrame([(117,119,116,118),(118,119,113,114),(114,115,109.5,112),(112,117.2,111.8,117)],
                       columns=["open","high","low","close"],index=pd.date_range("2026-10-04 06:55",periods=4,freq="5min",tz="UTC"))
    return {"1d":daily,"1h":hourly,"5m":pd.concat([warmup,signal])}


def _diagnose(frames=None,parameters=None,now=NOW):
    return diagnose_volium_v4_from_df(symbol="BTC_USDT",frames=frames or _frames(),now=now,parameters=parameters)


def _mirror(frames):
    return {key:pd.DataFrame({"open":250-frame.open,"high":250-frame.low,"low":250-frame.high,
                             "close":250-frame.close},index=frame.index) for key,frame in frames.items()}


@pytest.mark.parametrize("short",[False,True])
def test_complete_direction_without_pre_a_trend_or_right_confirmation_for_a(short):
    frames=_mirror(_frames()) if short else _frames()
    result=_diagnose(frames)
    assert result.setup is not None and result.setup.direction==("SHORT" if short else "LONG")
    context=v1._closed(frames["1d"],"1d",pd.Timestamp(NOW))
    points=v1._pivots(context,2)
    a_i=context.index.get_loc(pd.Timestamp(result.features["context_origin_open_utc"]))
    assert v1._trend([point for point in points if point.index+2<a_i]) is None
    assert not any(point.index==a_i and point.kind==("high" if short else "low") for point in points)
    assert result.features["pre_A_trend_required"] is False
    assert result.features["origin_requires_right_pivot_confirmation"] is False
    assert pd.Timestamp(result.features["context_confirmation_open_utc"])==pd.Timestamp(result.features["context_origin_open_utc"])+pd.Timedelta(days=1)
    assert pd.Timestamp(result.features["context_confirmation_known_close_utc"])<=pd.Timestamp(NOW)
    assert result.setup.take_profits==[120 if short else 130]
    assert result.setup.stop_loss==pytest.approx(140.5*1.0002 if short else 109.5*.9998)
    rr=abs(result.setup.take_profits[0]-result.setup.entry)/abs(result.setup.entry-result.setup.stop_loss)
    assert rr==pytest.approx(2)


def test_a_requires_actual_strict_sweep_not_only_a_low_and_strong_c():
    frames=_frames()
    frames["1d"].iloc[7]=[108,115,90,110]  # Only equals the known low90.
    result=_diagnose(frames)
    assert result.setup is None and result.reason=="no_actual_daily_sweep_reclaim"


def test_swept_level_must_be_known_before_a_even_though_a_needs_no_right_bars():
    context=_frames()["1d"].copy()
    context.iloc[6]=[108,116,88,109]  # This day consumes known low90.
    context.iloc[7]=[108,115,87,110]  # Low88 is not a confirmed preexisting pivot.
    leg=_post_sweep_break_leg(context,2)
    assert leg.origin_index==6  # Does not invent a new A from the unknown low88.
    assert leg.reason=="immediate_confirmation_not_directional"


def test_unclosed_future_c_cannot_confirm_and_mutating_it_has_no_effect():
    frames=_frames()
    for tf in ("1h","5m"):
        frames[tf].index=frames[tf].index-pd.Timedelta(days=1)
    now=datetime(2026,10,3,7,15,tzinfo=timezone.utc)
    expected=_diagnose(frames,now=now).as_dict()
    assert expected["reason"]=="immediate_confirmation_not_closed"
    frames["1d"].iloc[8]=[111,190,95,185]
    assert _diagnose(frames,now=now).as_dict()==expected


def test_only_immediate_next_day_c_is_allowed_not_a_later_break():
    context=_frames()["1d"].copy()
    context.iloc[8]=[111,119,95,114]  # C fails, without creating a new SHORT raid of high120.
    context.iloc[9]=[116,130,96,129]  # Following day succeeds but cannot rescue it.
    leg=_post_sweep_break_leg(context,2)
    assert leg.reason=="immediate_confirmation_not_directional"
    assert context.index[leg.origin_index]==pd.Timestamp("2026-10-02T00:00Z")


def test_missing_calendar_next_day_does_not_promote_next_available_row():
    context=_frames()["1d"].copy()
    context=context.drop(pd.Timestamp("2026-10-03T00:00Z"))
    context.iloc[-1]=[116,130,96,129]
    assert _post_sweep_break_leg(context,2).reason=="missing_immediate_confirmation_day"


@pytest.mark.parametrize("bad_c",[(111,119,95,115),(111,125,89,123),(111,115,95,115)])
def test_confirmation_equalities_do_not_pass_strict_break_or_hl(bad_c):
    frames=_frames()
    frames["1d"].iloc[8]=bad_c
    assert _diagnose(frames).reason=="immediate_confirmation_not_directional"


def _with_newer_a(context,include_bad_c):
    rows=[(118,122,97,119),(108,114,88,100)]
    if include_bad_c:
        rows.append((110,115,92,113))
    extra=pd.DataFrame(rows,columns=["open","high","low","close"],
        index=pd.date_range(context.index[-1]+pd.Timedelta(days=1),periods=len(rows),freq="D"))
    return pd.concat([context,extra])


@pytest.mark.parametrize("has_bad_c",[False,True])
def test_latest_actual_a_without_required_c_rejects_without_older_fallback(has_bad_c):
    context=_with_newer_a(_frames()["1d"],has_bad_c)
    leg=_post_sweep_break_leg(context,2)
    assert context.index[leg.origin_index]==pd.Timestamp("2026-10-06T00:00Z")
    assert leg.reason==("immediate_confirmation_not_directional" if has_bad_c else "immediate_confirmation_not_closed")


def test_latest_double_raid_is_ambiguous_even_if_next_day_breaks_up():
    context=_frames()["1d"].copy()
    context.iloc[7]=[108,141,89,110]  # Sweeps/reclaims both low90 and high140.
    context.iloc[8]=[142,145,95,144]
    leg=_post_sweep_break_leg(context,2)
    assert leg.reason=="ambiguous_double_daily_raid" and leg.direction is None
    assert leg.origin_index==7


def test_c_creating_a_new_opposite_raid_resets_latest_a_without_old_fallback():
    frames=_frames()
    frames["1d"].iloc[8]=[111,125,95,115]  # New SHORT sweep/reclaim of known high120.
    result=_diagnose(frames)
    assert result.reason=="immediate_confirmation_not_closed" and result.setup is None
    assert result.features["context_origin_open_utc"]=="2026-10-03T00:00:00+00:00"
    assert result.features["context_origin_level"]==120


@pytest.mark.parametrize("short",[False,True])
def test_b_equal_touch_in_closed_h1_consumes_target_in_forming_d1(short):
    frames=_frames()
    frames["1h"].iloc[-1]=[120,140,117,121]
    if short:
        frames=_mirror(frames)
    result=_diagnose(frames)
    assert result.setup is None and result.reason=="context_target_consumed_lower_tf"


@pytest.mark.parametrize("timeframe",["1h","5m"])
def test_a_integrity_in_closed_lower_tf_is_preserved(timeframe):
    frames=_frames()
    frames[timeframe].iloc[5]=[117.8,118.4,88,118]
    result=_diagnose(frames)
    assert result.setup is None and result.reason=="origin_extreme_broken_lower_tf"


def test_future_data_cannot_rewrite_a_c_b_or_entry_signal():
    frames=_frames()
    expected=_diagnose(frames).as_dict()
    assert expected["accepted"]
    for tf,frame in list(frames.items()):
        future=pd.DataFrame([(160,200,50,180)],columns=["open","high","low","close"],
            index=pd.DatetimeIndex(["2026-10-05T00:00Z"]))
        frames[tf]=pd.concat([frame,future])
    assert _diagnose(frames).as_dict()==expected


def test_context_cache_uses_content_not_mutable_frame_identity():
    frames=_frames()
    assert _diagnose(frames).setup is not None
    frames["1d"].iloc[8]=[111,119,95,115]
    assert _diagnose(frames).reason=="immediate_confirmation_not_directional"


def test_closed_d1_after_c_still_must_keep_a_extreme_intact():
    context=_frames()["1d"].copy()
    context.iloc[9]=[100,120,88,105]
    assert _post_sweep_break_leg(context,2).reason=="origin_extreme_broken"


def test_b_can_form_after_a_but_only_after_its_own_causal_confirmation():
    frames=_frames()
    frames["1d"].iloc[8]=[121,145,95,144]  # Consumes old140 without a SHORT reclaim.
    extra=pd.DataFrame([(116,119,96,117)],columns=["open","high","low","close"],
        index=pd.DatetimeIndex(["2026-10-05T00:00Z"]))
    frames["1d"]=pd.concat([frames["1d"],extra])
    early={tf:frame.copy() for tf,frame in frames.items()}
    for tf in ("1h","5m"):
        early[tf].index=early[tf].index+pd.Timedelta(days=1)
        frames[tf].index=frames[tf].index+pd.Timedelta(days=2)
    assert _diagnose(early,now=datetime(2026,10,5,7,15,tzinfo=timezone.utc)).reason=="no_current_leg_target"
    result=_diagnose(frames,now=datetime(2026,10,6,7,15,tzinfo=timezone.utc))
    assert result.setup is not None and result.features["context_target"]==145
    assert result.features["context_target_after_origin"] is True
    assert result.features["context_target_known_close_utc"]=="2026-10-06T00:00:00+00:00"


def test_completed_new_leg_guard_is_preserved():
    context=_frames()["1d"].copy()
    context.iloc[9]=[118,120,97,119]
    rows=[(119,122,99,120),(115,119,94,116),(118,124,98,122),
          (126,130,99,128),(121,124,98,120),(121,123,100,122)]
    extra=pd.DataFrame(rows,columns=["open","high","low","close"],
        index=pd.date_range(context.index[-1]+pd.Timedelta(days=1),periods=len(rows),freq="D"))
    leg=_post_sweep_break_leg(pd.concat([context,extra]),2)
    assert leg.reason=="completed_new_leg_after_origin"


@pytest.mark.parametrize("short",[False,True])
def test_batch_provider_matches_direct_for_the_only_24_strength_grid(short):
    parameters=[V4Parameters(reaction_min_atr=atr,reaction_min_body_ratio=body,
        reaction_min_prior_body_ratio=prior,reaction_max_bars=bars)
        for atr,body,prior,bars in product((.8,1.2,1.6),(.6,.7),(1.,1.5),(2,3))]
    frames=_mirror(_frames()) if short else _frames()
    kwargs=dict(symbol="BTC_USDT",frames=frames,now=NOW)
    direct=[analyze_volium_v4_from_df(**kwargs,parameters=p) for p in parameters]
    batch=analyze_volium_v4_batch_from_df(**kwargs,parameter_sets=parameters)
    assert all(x is not None for x in direct)
    assert [x.model_dump() for x in direct]==[x.model_dump() for x in batch]
    assert make_v4_analyzer(parameters[0])(**kwargs)==direct[0]


def test_raid_identity_actual_edge_and_original_body_still_match_v3_policy():
    frames=_frames()
    prior=_diagnose(frames).setup
    extra=pd.DataFrame([(112,119.2,109,119)],columns=["open","high","low","close"],
        index=pd.DatetimeIndex(["2026-10-04T07:15Z"]))
    frames["5m"]=pd.concat([frames["5m"],extra])
    result=_diagnose(frames,now=datetime(2026,10,4,7,20,tzinfo=timezone.utc))
    assert result.setup is not None and result.setup.id==prior.id
    assert result.setup.stop_loss==pytest.approx(109*.9998)
    assert result.features["raid_start_open_utc"]=="2026-10-04T07:05:00+00:00"
    assert result.features["opposite_body_open_utc"]=="2026-10-04T07:05:00+00:00"


@pytest.mark.parametrize("kwargs",[{"context_mode":"latest_leg_daily"},{"correction_target_mode":"last_pivot"},
    {"reaction_min_atr":float("nan")},{"reaction_atr_period":2.5},{"reaction_max_bars":0}])
def test_invalid_parameters_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        V4Parameters(**kwargs)
