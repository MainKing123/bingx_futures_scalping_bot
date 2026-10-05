from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import pytest

from app.strategy import volium_v2 as frozen_v2
from app.strategy.volium_v2_corrected import (diagnose_volium_v2_corrected_from_df,
    analyze_volium_v2_corrected_batch_from_df, make_v2_corrected_analyzer)
from app.strategy.volium_v3 import (V3Parameters, diagnose_volium_v3_from_df,
    analyze_volium_v3_from_df, analyze_volium_v3_batch_from_df, make_v3_analyzer)


NOW = datetime(2026,10,5,7,15,tzinfo=timezone.utc)


def _frame(highs,lows,end,freq):
    index = pd.date_range(end=end,periods=len(highs),freq=freq,tz="UTC")
    middle = [(high+low)/2 for high,low in zip(highs,lows)]
    return pd.DataFrame({"open":middle,"high":highs,"low":lows,"close":middle},index=index)


def _frames():
    daily = _frame([100,102,103,110,102,104,106,125,110,112,116,140,118,120,122,124,119,118],
                   [90,89,80,90,90,90,79,100,95,100,94,108,108,109,93,116,115,116],"2026-10-04","D")
    daily.iloc[14] = [112,122,93,118]
    hourly = _frame([119,121,118,125,121,120,118,116,130,120,121,122,123],
                    [115,112,105,113,113,114,113,110,115,115,116,116,117],"2026-10-05 06:00","h")
    warmup = pd.DataFrame([(117.8,118.4,117.4,118.)]*16,columns=["open","high","low","close"],
                          index=pd.date_range("2026-10-05 05:35",periods=16,freq="5min",tz="UTC"))
    signal = pd.DataFrame([(117,119,116,118),(118,119,113,114),(114,115,109.5,112),(112,117.2,111.8,117)],
                         columns=["open","high","low","close"],
                         index=pd.date_range("2026-10-05 06:55",periods=4,freq="5min",tz="UTC"))
    return {"1d":daily,"1h":hourly,"5m":pd.concat([warmup,signal])}


def _local_frames():
    frames = _frames()
    # Daily A is HL95 above the prior daily HL94. It does not sweep D1-low94.
    frames["1d"].iloc[14] = [112,122,95,118]
    prefix_index = pd.date_range(pd.Timestamp("2026-09-27T00:00Z"),frames["1h"].index[0]-pd.Timedelta(hours=1),freq="h")
    prefix = pd.DataFrame([(115,119,110,115)]*len(prefix_index),columns=["open","high","low","close"],index=prefix_index)
    prefix.loc[pd.Timestamp("2026-09-30T18:00Z")] = [115,119,100,115]
    prefix.loc[pd.Timestamp("2026-10-01T03:00Z")] = [106,113,95,111]
    frames["1h"] = pd.concat([prefix,frames["1h"]])
    return frames


def _diagnose(frames=None,parameters=None,**kwargs):
    return diagnose_volium_v3_from_df(symbol="BTC_USDT",frames=frames or _frames(),now=NOW,parameters=parameters,**kwargs)


def _mirror(frames):
    return {key:pd.DataFrame({"open":250-frame.open,"high":250-frame.low,
        "low":250-frame.high,"close":250-frame.close},index=frame.index) for key,frame in frames.items()}


def test_complete_daily_long_keeps_structure_v_and_exact_2r():
    result = _diagnose()
    assert result.setup is not None and result.reason == "accepted"
    assert result.setup.direction == "LONG" and result.setup.take_profits == [130]
    assert result.setup.stop_loss == pytest.approx(109.5*.9998)
    assert (130-result.setup.entry)/(result.setup.entry-result.setup.stop_loss) == pytest.approx(2)
    assert result.features["context_target"] == 140
    assert result.features["targets_consumed_lower_tf"] >= 1


def test_b_after_a_is_known_causally_and_creation_does_not_consume_it():
    frames = _frames()
    frames["1d"].iloc[14] = [112,141,93,118]  # Original pre-A B140 is consumed.
    frames["1d"].iloc[15] = [120,142,116,121]  # New B confirms only at Oct5 00Z.
    result = _diagnose(frames)
    assert result.setup is not None and result.features["context_target"] == 142
    assert result.features["context_target_after_origin"] is True
    assert pd.Timestamp(result.features["context_target_known_close_utc"]) <= pd.Timestamp(NOW)
    assert frozen_v2.diagnose_volium_v2_from_df(symbol="BTC_USDT",frames=frames,now=NOW).setup is None
    # Removing one right-hand confirmation must not retroactively create B.
    frames["1d"] = frames["1d"].iloc[:-1]
    assert _diagnose(frames).setup is None


@pytest.mark.parametrize("local",[False,True])
def test_short_is_symmetric_for_both_context_hypotheses(local):
    parameters = V3Parameters(context_mode="latest_leg_local" if local else "latest_leg_daily")
    result = _diagnose(_mirror(_local_frames() if local else _frames()),parameters)
    assert result.setup is not None and result.setup.direction == "SHORT"
    assert result.setup.take_profits == [120]
    assert (result.setup.entry-120)/(result.setup.stop_loss-result.setup.entry) == pytest.approx(2)


def test_local_hl_has_actual_causal_h1_sweep_not_a_fabricated_daily_sweep():
    frames = _local_frames()
    result = _diagnose(frames,V3Parameters(context_mode="latest_leg_local"))
    assert result.setup is not None
    assert result.features["context_origin_extreme"] == 95
    assert result.features["context_origin_level"] == 100
    assert result.features["origin_evidence_tf"] == "1h"
    assert result.features["origin_sweep_open_utc"] == "2026-10-01T03:00:00+00:00"
    assert pd.Timestamp(result.features["origin_swept_level_open_utc"]) < pd.Timestamp(result.features["origin_sweep_open_utc"])


def test_hl_without_any_sweep_is_ineligible():
    frames = _local_frames()
    frames["1h"].loc[pd.Timestamp("2026-09-30T18:00Z")] = [115,119,110,115]
    result = _diagnose(frames,V3Parameters(context_mode="latest_leg_local"))
    assert result.setup is None and result.reason == "local_origin_without_actual_sweep"


def test_missing_old_h1_evidence_is_reported_and_not_invented():
    frames = _local_frames()
    frames["1h"] = frames["1h"].tail(110)
    result = _diagnose(frames,V3Parameters(context_mode="latest_leg_local"))
    assert result.setup is None and result.reason == "insufficient_origin_liquidity_history"


def _forming_day_breach_frames():
    frames = _frames()
    frames["1d"].iloc[-1] = [120,125,105,116]
    rows = [(117,119,115,118),(118,121,112,116.5),(110,118,105,111.5),
            (120,125,113,121),(118,121,113,117),(118,120,114,117),(115,118,113,116),
            (113,116,110,114),(119,124,115,120),(118,120,115,117.5),(119,121,116,118),
            (115,120,92,118),(118,122,113,120),(120,123,114,122),(116,121,110,115),
            (120,130,115,122),(120,124,116,121),(121,123,117,120)]
    frames["1h"] = pd.DataFrame(rows,columns=["open","high","low","close"],
        index=pd.date_range(end="2026-10-05 06:00",periods=18,freq="h",tz="UTC"))
    return frames


def test_closed_h1_breach_in_forming_daily_invalidates_a_and_corrected_control():
    frames = _forming_day_breach_frames()
    kwargs = dict(symbol="BTC_USDT",frames=frames,now=NOW)
    assert frozen_v2.analyze_volium_v2_from_df(**kwargs) is not None  # Preserve old control.
    v3 = diagnose_volium_v3_from_df(**kwargs)
    corrected = diagnose_volium_v2_corrected_from_df(**kwargs)
    assert v3.setup is None and corrected.setup is None
    assert v3.reason == corrected.reason == "origin_extreme_broken_lower_tf"


def test_unclosed_lower_tf_breach_cannot_invalidate_a_early():
    frames = _frames()
    frames["1h"] = pd.concat([frames["1h"],pd.DataFrame([(115,120,92,118)],
        columns=["open","high","low","close"],index=pd.DatetimeIndex(["2026-10-05T07:00Z"]))])
    assert _diagnose(frames).setup is not None
    assert diagnose_volium_v2_corrected_from_df(symbol="BTC_USDT",frames=frames,now=NOW).setup is not None


def test_closed_m5_breach_also_invalidates_a_without_waiting_for_h1_or_d1():
    frames = _frames()
    frames["5m"].iloc[5] = [117.8,118.4,92,118]
    kwargs = dict(symbol="BTC_USDT",frames=frames,now=NOW)
    assert frozen_v2.analyze_volium_v2_from_df(**kwargs) is not None
    assert diagnose_volium_v3_from_df(**kwargs).reason == "origin_extreme_broken_lower_tf"
    assert diagnose_volium_v2_corrected_from_df(**kwargs).reason == "origin_extreme_broken_lower_tf"


@pytest.mark.parametrize("local",[False,True])
def test_future_candles_and_future_pivot_confirmation_do_not_change_signal(local):
    frames = _local_frames() if local else _frames()
    parameters = V3Parameters(context_mode="latest_leg_local" if local else "latest_leg_daily")
    expected = _diagnose(frames,parameters).as_dict()
    assert expected["accepted"]
    for key,frame in list(frames.items()):
        future = pd.DataFrame([(160,200,50,180)],columns=["open","high","low","close"],
            index=pd.DatetimeIndex(["2026-10-06T00:00Z"]))
        frames[key] = pd.concat([frame,future])
    assert _diagnose(frames,parameters).as_dict() == expected
    assert analyze_volium_v3_from_df(symbol="BTC_USDT",frames=frames,parameters=parameters,
        now=datetime(2026,10,5,7,14,tzinfo=timezone.utc)) is None


@pytest.mark.parametrize("local",[False,True])
def test_batch_direct_and_provider_match_all_strength_variants_and_contexts(local):
    from itertools import product
    parameters = [V3Parameters(context_mode=context,reaction_min_atr=atr,reaction_min_body_ratio=body,
        reaction_min_prior_body_ratio=prior,reaction_max_bars=bars)
        for context,atr,body,prior,bars in product(("latest_leg_daily","latest_leg_local"),(.8,1.2,1.6),(.6,.7),(1.,1.5),(2,3))]
    kwargs = dict(symbol="BTC_USDT",frames=_local_frames() if local else _frames(),now=NOW)
    expected = [analyze_volium_v3_from_df(**kwargs,parameters=p) for p in parameters]
    actual = analyze_volium_v3_batch_from_df(**kwargs,parameter_sets=parameters)
    assert [x.model_dump() if x else None for x in actual] == [x.model_dump() if x else None for x in expected]
    assert make_v3_analyzer(parameters[0])(**kwargs) == expected[0]


def test_shallow_repeat_is_one_raid_with_original_stop_and_strength_anchor():
    frames = _frames()
    frames["5m"] = pd.concat([frames["5m"],pd.DataFrame([(112,119.2,109.8,119)],
        columns=["open","high","low","close"],index=pd.DatetimeIndex(["2026-10-05T07:15Z"]))])
    kwargs = dict(symbol="BTC_USDT",frames=frames,now=datetime(2026,10,5,7,20,tzinfo=timezone.utc))
    broad = replace(V3Parameters(),reaction_min_atr=0,reaction_min_prior_body_ratio=0)
    candidates = diagnose_volium_v3_from_df(**kwargs,parameters=broad,_collect_candidates=True).features["structural_candidates"]
    assert len(candidates) == 1
    assert candidates[0].features["raid_start_open_utc"] == "2026-10-05T07:05:00+00:00"
    assert candidates[0].features["sweep_open_utc"] == "2026-10-05T07:05:00+00:00"
    assert candidates[0].setup.stop_loss == pytest.approx(109.5*.9998)
    assert candidates[0].features["opposite_body_open_utc"] == "2026-10-05T07:05:00+00:00"
    strict = replace(V3Parameters(),reaction_min_atr=candidates[0].features["confirm_body_atr"]+.1)
    expected = analyze_volium_v3_from_df(**kwargs,parameters=strict)
    actual = analyze_volium_v3_batch_from_df(**kwargs,parameter_sets=[broad,strict])
    assert expected is None and actual[1] is None
    assert actual[0].id == _diagnose().setup.id


@pytest.mark.parametrize("short",[False,True])
def test_deepening_raid_uses_actual_extreme_first_id_and_preserves_earlier_object(short):
    frames = _frames()
    if short:
        frames = _mirror(frames)
    prior = _diagnose(frames)
    frozen_prior = prior.as_dict()
    new = pd.DataFrame([(112,119.2,109,119)],columns=["open","high","low","close"],
        index=pd.DatetimeIndex(["2026-10-05T07:15Z"]))
    if short:
        new = _mirror({"5m":new})["5m"]
    frames["5m"] = pd.concat([frames["5m"],new])
    kwargs = dict(symbol="BTC_USDT",frames=frames,now=datetime(2026,10,5,7,20,tzinfo=timezone.utc))
    result = diagnose_volium_v3_from_df(**kwargs)
    assert result.setup is not None and result.setup.id == prior.setup.id
    assert result.features["raid_start_open_utc"] == "2026-10-05T07:05:00+00:00"
    assert result.features["sweep_open_utc"] == "2026-10-05T07:15:00+00:00"
    assert result.features["reaction_age_bars"] == 2
    assert result.features["atr_last_open_utc"] == "2026-10-05T07:00:00+00:00"
    assert result.setup.stop_loss == pytest.approx(141*1.0002 if short else 109*.9998)
    assert prior.as_dict() == frozen_prior
    assert _diagnose(frames).as_dict() == frozen_prior  # New candle is unclosed at prior timestamp.


@pytest.mark.parametrize("short",[False,True])
def test_closed_reclaim_then_doji_repeat_never_replaces_original_body_or_id(short):
    frames = _frames()
    prior = _diagnose(frames)
    extra = pd.DataFrame([(117.2,118,109.8,117.1),(117.1,123.2,116.9,123)],
        columns=["open","high","low","close"],
        index=pd.date_range("2026-10-05T07:15Z",periods=2,freq="5min"))
    frames["5m"] = pd.concat([frames["5m"],extra])
    if short:
        frames = _mirror(frames)
        prior = _diagnose(_mirror(_frames()))
    kwargs = dict(symbol="BTC_USDT",frames=frames,now=datetime(2026,10,5,7,25,tzinfo=timezone.utc))
    result = diagnose_volium_v3_from_df(**kwargs)
    assert result.setup is not None and result.setup.id == prior.setup.id
    assert result.features["reaction_age_bars"] == 3
    assert result.features["opposite_body"] == pytest.approx(2)
    assert result.features["confirm_body_prior_body_ratio"] == pytest.approx(2.95)
    assert result.setup.stop_loss == prior.setup.stop_loss
    # A new late confirmation cannot reset max-wait by using the shallow doji.
    late = pd.DataFrame([(123,125.2,122.8,125)],columns=["open","high","low","close"],
        index=pd.DatetimeIndex(["2026-10-05T07:25Z"]))
    if short:
        late = _mirror({"5m":late})["5m"]
    frames["5m"] = pd.concat([frames["5m"],late])
    assert diagnose_volium_v3_from_df(**{**kwargs,"now":datetime(2026,10,5,7,30,tzinfo=timezone.utc)}).setup is None
    variants = [V3Parameters(),replace(V3Parameters(),reaction_max_bars=2),
        replace(V3Parameters(),reaction_min_prior_body_ratio=3)]
    direct = [analyze_volium_v3_from_df(**kwargs,parameters=p) for p in variants]
    batch = analyze_volium_v3_batch_from_df(**kwargs,parameter_sets=variants)
    assert [x.model_dump() if x else None for x in direct] == [x.model_dump() if x else None for x in batch]


def test_corrected_control_preserves_intact_geometry_and_batch():
    kwargs = dict(symbol="BTC_USDT",frames=_frames(),now=NOW)
    original = frozen_v2.analyze_volium_v2_from_df(**kwargs)
    corrected = make_v2_corrected_analyzer()(**kwargs)
    assert corrected == original
    assert analyze_volium_v2_corrected_batch_from_df(**kwargs,parameter_sets=[frozen_v2.V2Parameters()]) == [corrected]


@pytest.mark.parametrize("kwargs",[{"context_mode":"legacy_context"},{"correction_target_mode":"last_pivot"},
    {"reaction_min_atr":float("nan")},{"reaction_atr_period":2.5},{"reaction_max_bars":0}])
def test_invalid_parameters_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        V3Parameters(**kwargs)


def test_v3_does_not_invent_new_scalp_or_swing_contexts():
    with pytest.raises(ValueError,match="only for intraday"):
        _diagnose(mode="scalp")
