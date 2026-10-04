from dataclasses import replace
from datetime import datetime,timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from app.strategy.volium import analyze_volium_from_df
from app.strategy.volium_v2 import (V2Parameters, _preexisting_leg, analyze_volium_v2_from_df,
    analyze_volium_v2_batch_from_df, diagnose_volium_v2_from_df, make_v2_analyzer)


NOW = datetime(2026,10,5,7,15,tzinfo=timezone.utc)


def _frame(highs,lows,end,freq):
    index = pd.date_range(end=end,periods=len(highs),freq=freq,tz="UTC")
    middle = [(high+low)/2 for high,low in zip(highs,lows)]
    return pd.DataFrame({"open":middle,"high":highs,"low":lows,"close":middle},index=index)


def _fixtures():
    daily = _frame([100,102,103,110,102,104,106,125,110,112,116,140,118,120,122,124,119,118],
                   [90,89,80,90,90,90,79,100,95,100,94,108,108,109,93,116,115,116],"2026-10-04","D")
    daily.iloc[14] = [112,122,93,118]  # Fresh A reclaims94; B140 was known before A.
    hourly = _frame([119,121,118,125,121,120,118,116,130,120,121,122,123],
                    [115,112,105,113,113,114,113,110,115,115,116,116,117],"2026-10-05 06:00","h")
    warmup = pd.DataFrame([(117.8,118.4,117.4,118.)]*16,columns=["open","high","low","close"],
                          index=pd.date_range("2026-10-05 05:35",periods=16,freq="5min",tz="UTC"))
    signal = pd.DataFrame([(117,119,116,118),(118,119,113,114),(114,115,109.5,112),(112,117.2,111.8,117)],
                         columns=["open","high","low","close"],
                         index=pd.date_range("2026-10-05 06:55",periods=4,freq="5min",tz="UTC"))
    return {"1d":daily,"1h":hourly,"5m":pd.concat([warmup,signal])}


def _diagnose(frames=None,parameters=None,**kwargs):
    return diagnose_volium_v2_from_df(symbol="BTC_USDT",frames=frames or _fixtures(),now=NOW,parameters=parameters,**kwargs)


def test_complete_long_links_fresh_a_to_preexisting_b_and_keeps_fixed_2r():
    frames = _fixtures()
    result = _diagnose(frames)
    assert result.reason == "accepted" and result.setup is not None
    setup = result.setup
    assert setup.direction == "LONG" and setup.take_profits == [130]
    assert setup.stop_loss == pytest.approx(109.5*.9998)
    assert (130-setup.entry)/(setup.entry-setup.stop_loss) == pytest.approx(2)
    assert result.features["context_target"] == 140
    assert pd.Timestamp(result.features["context_target_open_utc"]) < pd.Timestamp(result.features["context_origin_open_utc"])
    assert pd.Timestamp(result.features["atr_last_open_utc"]) < pd.Timestamp(result.features["sweep_open_utc"])
    # The reclaimed sweep is a lower wick, but the causal pre-A trend is HH/HL.
    assert analyze_volium_from_df(symbol="BTC_USDT",frames=frames,now=NOW) is None


def test_complete_short_is_symmetric():
    frames = _fixtures()
    for key,frame in list(frames.items()):
        frames[key] = pd.DataFrame({"open":250-frame.open,"high":250-frame.low,
                                   "low":250-frame.high,"close":250-frame.close},index=frame.index)
    result = _diagnose(frames)
    assert result.setup is not None and result.setup.direction == "SHORT"
    assert result.setup.take_profits == [120]
    assert (result.setup.entry-120)/(result.setup.stop_loss-result.setup.entry) == pytest.approx(2)


def test_scalp_keeps_active_h1_and_m5_target_without_invented_h1_origin_requirement():
    frames = _fixtures()
    hourly = frames["1h"].copy()
    hourly.iloc[-3] = [115,122,114,120]
    hourly.iloc[-2] = [120,124,118,123]
    hourly.iloc[-1] = [123,129,122,128]
    five = frames["1h"].copy()
    five.index = pd.date_range("2026-10-05 05:55",periods=len(five),freq="5min",tz="UTC")
    minute = frames["5m"].copy()
    minute.index = pd.date_range("2026-10-05 06:55",periods=len(minute),freq="min",tz="UTC")
    result = _diagnose({"1h":hourly,"5m":five,"1m":minute},mode="scalp",
        settings=SimpleNamespace(volium_active_trend_min_atr=.5))
    assert result.setup is not None and result.setup.setup_type == "VOLIUM_SCALP"
    assert result.setup.take_profits == [130]
    assert result.setup.stop_loss == pytest.approx(109.5*.999)
    assert result.features["context_method"] == "H1 trend and active progress"


def test_future_context_entry_and_pivots_do_not_change_closed_signal():
    frames = _fixtures()
    expected = _diagnose(frames).as_dict()
    for key,frame in list(frames.items()):
        future = pd.DataFrame([(160,200,50,180)],columns=["open","high","low","close"],
                              index=pd.DatetimeIndex(["2026-10-06T00:00Z"]))
        frames[key] = pd.concat([frame,future])
    assert _diagnose(frames).as_dict() == expected
    assert analyze_volium_v2_from_df(symbol="BTC_USDT",frames=frames,now=datetime(2026,10,5,7,14,tzinfo=timezone.utc)) is None


def test_old_unrelated_a_does_not_authorize_a_completed_new_leg():
    frames = _fixtures()
    extra = pd.DataFrame({"high":[121,120,122,123,121,120],"low":[112,109,112,111,112,113]})
    extra["open"] = extra["close"] = (extra.high+extra.low)/2
    daily = pd.concat([frames["1d"].reset_index(drop=True),extra],ignore_index=True)
    daily.index = pd.date_range(end="2026-10-04",periods=len(daily),freq="D",tz="UTC")
    frames["1d"] = daily
    result = _diagnose(frames)
    assert result.setup is None and result.reason == "completed_new_leg_after_origin"


def test_b_must_be_confirmed_before_a_not_created_by_later_future_pivots():
    daily = _fixtures()["1d"]
    # B140 is consumed by A's upper range; a newly formed later high142 cannot
    # replace it retroactively as the preexisting destination of this leg.
    daily.iloc[14] = [112,141,93,118]
    daily.iloc[15] = [120,142,116,121]
    leg = _preexisting_leg(daily,2)
    assert leg.reason == "preexisting_target_consumed"


def test_pre_a_peak_whose_right_confirmation_arrives_after_a_is_not_a_known_b():
    daily = _fixtures()["1d"]
    daily.iloc[11] = [116,130,108,120]
    daily.iloc[13] = [120,140,109,124]
    # Peak13 exists physically before A14, but it is only confirmed by bars
    # 14/15. Older high125 is already consumed; no known B remains before A.
    assert _preexisting_leg(daily,2).reason == "preexisting_target_consumed"


def test_target_touched_in_forming_daily_candle_is_not_treated_as_unhit():
    frames = _fixtures()
    frames["1h"].iloc[-1] = [120,141,117,121]
    result = _diagnose(frames)
    assert result.setup is None and result.reason == "context_target_consumed_lower_tf"


def test_atr_strength_is_relative_to_previous_candles_not_just_body_fraction():
    frames = _fixtures()
    frames["5m"].iloc[:16] = [117.8,123,112,118]
    weak = _diagnose(frames)
    assert weak.reason == "reaction_below_atr" and weak.setup is None
    assert weak.features["confirm_body_fraction"] > .9
    assert weak.features["confirm_body_atr"] < .8
    permissive = _diagnose(frames,replace(V2Parameters(),reaction_min_atr=.3))
    assert permissive.setup is not None


def test_final_body_must_be_strong_relative_to_opposite_body():
    result = _diagnose(parameters=replace(V2Parameters(),reaction_min_prior_body_ratio=3))
    assert result.setup is None and result.reason == "reaction_below_prior_body"
    assert result.features["confirm_body_prior_body_ratio"] == 2.5


def test_structural_origin_is_not_a_minor_lower_high_inside_correction():
    frames = _fixtures()
    hourly = frames["1h"]
    prefix = hourly.iloc[:9].copy()
    prefix.index = pd.date_range(end="2026-10-05 00:00",periods=9,freq="h",tz="UTC")
    extra = pd.DataFrame([(120,120,115,119),(119,121,116,120),(120,120,115,119),
                          (119,124,116,121),(121,122,116,120),(120,121,117,119)],
                         columns=["open","high","low","close"],
                         index=pd.date_range("2026-10-05 01:00",periods=6,freq="h",tz="UTC"))
    frames["1h"] = pd.concat([prefix,extra])
    original = _diagnose(frames)
    minor = _diagnose(frames,replace(V2Parameters(),correction_target_mode="last_pivot"))
    assert original.setup is not None and original.setup.take_profits == [130]
    assert minor.setup is not None and minor.setup.take_profits == [124]


def test_factory_is_compatible_and_batch_preserves_exact_variants():
    parameters = [V2Parameters(),replace(V2Parameters(),reaction_min_atr=4),
                  replace(V2Parameters(),reaction_min_prior_body_ratio=3),
                  replace(V2Parameters(),reaction_min_body_ratio=.99),
                  replace(V2Parameters(),reaction_max_bars=2)]
    kwargs = dict(symbol="BTC_USDT",frames=_fixtures(),now=NOW)
    expected = [analyze_volium_v2_from_df(**kwargs,parameters=parameter) for parameter in parameters]
    actual = analyze_volium_v2_batch_from_df(**kwargs,parameter_sets=parameters)
    assert [setup.model_dump() if setup else None for setup in actual] == [setup.model_dump() if setup else None for setup in expected]
    assert make_v2_analyzer(parameters[0])(**kwargs).model_dump() == expected[0].model_dump()


def test_batch_keeps_older_candidate_when_newest_sweep_fails_stricter_atr():
    frames = _fixtures()
    frames["5m"] = pd.concat([frames["5m"],pd.DataFrame([(112,119.2,109.8,119)],
        columns=["open","high","low","close"],index=pd.DatetimeIndex(["2026-10-05T07:15Z"]))])
    kwargs = dict(symbol="BTC_USDT",frames=frames,now=datetime(2026,10,5,7,20,tzinfo=timezone.utc))
    broad = replace(V2Parameters(),reaction_min_atr=0,reaction_min_prior_body_ratio=0)
    extracted = diagnose_volium_v2_from_df(**kwargs,parameters=broad,_collect_candidates=True)
    candidates = extracted.features["structural_candidates"]
    assert len(candidates) == 2
    newer_strength = candidates[0].features["confirm_body_atr"]
    older_strength = candidates[1].features["confirm_body_atr"]
    assert older_strength > newer_strength
    strict = replace(V2Parameters(),reaction_min_atr=(newer_strength+older_strength)/2)
    expected = analyze_volium_v2_from_df(**kwargs,parameters=strict)
    actual = analyze_volium_v2_batch_from_df(**kwargs,parameter_sets=[broad,strict])
    assert actual[0].id != expected.id
    assert actual[1].model_dump() == expected.model_dump()
    assert expected.stop_loss == pytest.approx(109.5*.9998)


def test_linked_context_cache_invalidates_after_same_object_origin_mutation():
    frames = _fixtures()
    assert _diagnose(frames).setup is not None
    frames["1d"].iloc[14] = [112,122,93,93.5]
    assert _diagnose(frames).setup is None


@pytest.mark.parametrize("kwargs",[{"reaction_min_atr":float("nan")},{"reaction_atr_period":1},
                                 {"context_mode":"invented"},{"reaction_max_bars":0},{"reaction_atr_period":2.5}])
def test_invalid_experimental_parameters_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        V2Parameters(**kwargs)
