"""Causal OHLC contract tests for the isolated, source-derived v5 hypothesis."""
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from app.strategy import volium as v1
from app.strategy.volium_v5 import (
    V5Parameters,
    analyze_volium_v5_batch_from_df,
    analyze_volium_v5_from_df,
    diagnose_volium_v5_batch_from_df,
    diagnose_volium_v5_from_df,
    make_v5_analyzer,
)


NOW = datetime(2026, 10, 5, 7, 15, tzinfo=timezone.utc)


def _frame(highs, lows, end, frequency):
    middle = [(high + low) / 2 for high, low in zip(highs, lows)]
    return pd.DataFrame(
        {"open": middle, "high": highs, "low": lows, "close": middle},
        index=pd.date_range(end=end, periods=len(highs), freq=frequency, tz="UTC"),
        dtype=float,
    )


def _frames():
    # Existing v1 context semantics: HH/HL and an intact actual D1 sweep/reclaim.
    daily = _frame(
        [100, 102, 103, 110, 102, 104, 106, 125, 110, 112, 116, 130, 118, 120, 122, 140, 125, 126],
        [90, 89, 80, 90, 90, 90, 79, 100, 95, 100, 94, 108, 108, 109, 105, 116, 115, 116],
        "2026-10-04", "D",
    )
    hourly = _frame(
        [119, 121, 118, 125, 121, 120, 118, 116, 130, 120, 121, 122, 123],
        [115, 112, 105, 113, 113, 114, 113, 110, 115, 115, 116, 116, 117],
        "2026-10-05 06:00", "h",
    )
    warmup = pd.DataFrame(
        [(117.8, 118.4, 117.4, 118.)] * 16,
        columns=["open", "high", "low", "close"], dtype=float,
        index=pd.date_range("2026-10-05 05:35", periods=16, freq="5min", tz="UTC"),
    )
    signal = pd.DataFrame(
        [(117, 119, 116, 118), (118, 119, 113, 114),
         (114, 115, 109.5, 112), (112, 117.2, 111.8, 117)],
        columns=["open", "high", "low", "close"], dtype=float,
        index=pd.date_range("2026-10-05 06:55", periods=4, freq="5min", tz="UTC"),
    )
    return {"1d": daily, "1h": hourly, "5m": pd.concat([warmup, signal])}


def _mirror(frames):
    return {
        key: pd.DataFrame(
            {"open": 250 - frame.open, "high": 250 - frame.low,
             "low": 250 - frame.high, "close": 250 - frame.close}, index=frame.index,
        )
        for key, frame in frames.items()
    }


def _diagnose(frames=None, **kwargs):
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("enforce_session_filter", False)
    return diagnose_volium_v5_from_df(
        symbol="BTC_USDT", frames=_frames() if frames is None else frames, **kwargs,
    )


def _append_entry(frames, rows):
    entry = frames["5m"]
    extra = pd.DataFrame(
        rows, columns=["open", "high", "low", "close"], dtype=float,
        index=pd.date_range(entry.index[-1] + pd.Timedelta(minutes=5), periods=len(rows), freq="5min"),
    )
    frames["5m"] = pd.concat([entry, extra])
    return (extra.index[-1] + pd.Timedelta(minutes=5)).to_pydatetime()


@pytest.mark.parametrize("short", [False, True])
def test_directional_engulf_with_daily_origin_preserves_fixed_structural_2r(short):
    frames = _mirror(_frames()) if short else _frames()
    context = v1._closed(frames["1d"], "1d", pd.Timestamp(NOW))
    direction = "SHORT" if short else "LONG"
    assert v1._trend(v1._pivots(context, 2)) == direction
    assert v1._origin_sweep(context, direction, 2)
    result = _diagnose(frames)
    assert result.setup is not None
    assert result.reason == "accepted"
    assert result.setup.direction == direction
    assert result.setup.take_profits == [120 if short else 130]
    assert result.setup.stop_loss == pytest.approx(140.5 * 1.0002 if short else 109.5 * .9998)
    reward = abs(result.setup.take_profits[0] - result.setup.entry)
    risk = abs(result.setup.entry - result.setup.stop_loss)
    assert reward / risk == pytest.approx(2)
    assert result.setup.timestamp == NOW
    assert result.as_dict()["accepted"] is True


def test_daily_origin_remains_v1_rule_and_can_only_be_explicitly_disabled():
    frames = _frames()
    frames["1d"].iloc[6, frames["1d"].columns.get_loc("low")] = 82
    assert not v1._origin_sweep(frames["1d"], "LONG", 2)
    assert _diagnose(frames).setup is None
    assert _diagnose(frames, settings=SimpleNamespace(volium_require_daily_origin_sweep=False)).setup is not None


def test_intraday_requires_h1_bias_to_agree_with_d1():
    frames = _frames()
    frames["1h"] = _mirror({"1h": frames["1h"]})["1h"]
    assert v1._trend(v1._pivots(frames["1h"], 2)) == "SHORT"
    assert _diagnose(frames).setup is None


@pytest.mark.parametrize("short", [False, True])
def test_one_third_is_inclusive_and_body_fraction_or_atr_are_not_extra_filters(short):
    frames = _frames()
    frames["5m"].iloc[-2] = [114, 116.5, 109, 112]
    # Exactly 7/21 recovered. The unusually tall wick makes the body fraction weak.
    frames["5m"].iloc[-1] = [112, 129, 111.8, 116]
    if short:
        frames = _mirror(frames)
    result = _diagnose(frames)
    assert result.setup is not None
    assert result.setup.take_profits == [120 if short else 130]
    frames["5m"].iloc[-1, frames["5m"].columns.get_loc("close")] = 134.0001 if short else 115.9999
    assert _diagnose(frames).setup is None


@pytest.mark.parametrize("short", [False, True])
def test_fvg_inversion_can_confirm_without_engulfing_the_opposite_body(short):
    frames = _frames()
    # The original third candle creates bearish FVG 115..116 at the actual low.
    # 114.5..117 does not cover the final opposite body 112..114.
    frames["5m"].iloc[-1] = [114.5, 117.2, 111.8, 117]
    if short:
        frames = _mirror(frames)
    assert _diagnose(frames).setup is not None


def test_without_engulf_or_an_eligible_fvg_recovery_alone_is_insufficient():
    frames = _frames()
    frames["5m"].iloc[-2] = [114, 116.5, 109.5, 112]  # No bearish FVG.
    frames["5m"].iloc[-1] = [114.5, 117.2, 111.8, 117]
    assert _diagnose(frames).setup is None


def test_engulfment_alone_is_sufficient_without_a_gap():
    frames = _frames()
    frames["5m"].iloc[-2] = [114, 116.5, 109.5, 112]
    result = _diagnose(frames)
    assert result.setup is not None
    assert result.features["confirmation_model"] == "engulfment"


def test_early_weak_inversion_can_wait_for_recovery_within_the_same_raid():
    frames = _frames()
    frames["5m"].iloc[-1] = [114.5, 116.2, 111.8, 116.1]
    early = _diagnose(frames)
    assert early.setup is None
    assert early.features["recovery_fraction"] < 1 / 3
    now = _append_entry(frames, [(114.5, 117.2, 111.8, 117)])
    later = _diagnose(frames, now=now)
    assert later.setup is not None
    assert later.features["confirmation_model"] == "fvg_inversion"
    assert later.features["raid_start_open_utc"] == early.features["raid_start_open_utc"]
    assert later.features["manipulation_origin"] == 130
    assert later.features["manipulation_span"] == pytest.approx(130 - 109.5)
    assert later.features["recovery_fraction"] == pytest.approx((117 - 109.5) / (130 - 109.5))


def test_fvg_already_inverted_before_the_current_raid_cannot_be_reused():
    frames = _frames()
    frames["5m"].iloc[9] = [117, 118, 116, 116.5]
    frames["5m"].iloc[10] = [115, 115.5, 114, 115]
    # A bearish gap is born at 06:25, then inverted by the normal 06:30 close118.
    frames["5m"].iloc[-2] = [114, 116.5, 109.5, 112]
    frames["5m"].iloc[-1] = [114.5, 117.2, 111.8, 117]
    assert _diagnose(frames).setup is None


def test_closes_must_be_strictly_monotone_from_the_actual_extreme():
    frames = _frames()
    now = _append_entry(frames, [(117, 118, 111.8, 116.5), (112, 120, 111.8, 119)])
    assert _diagnose(frames, now=now).setup is None


@pytest.mark.parametrize("short", [False, True])
def test_shallower_repeat_keeps_whole_raid_stop_and_stable_first_breach_identity(short):
    frames = _frames()
    if short:
        frames = _mirror(frames)
    first = _diagnose(frames).setup
    assert first is not None
    row = (138, 140.2, 130.6, 131) if short else (112, 119.4, 109.8, 119)
    now = _append_entry(frames, [row])
    second = _diagnose(frames, now=now).setup
    assert second is not None
    assert second.id == first.id
    assert second.stop_loss == first.stop_loss
    assert second.entry == first.entry


def test_deeper_repeat_updates_whole_extreme_without_changing_raid_identity():
    frames = _frames()
    first = _diagnose(frames).setup
    now = _append_entry(frames, [(112, 119.4, 109, 119)])
    second = _diagnose(frames, now=now).setup
    assert first is not None and second is not None
    assert second.id == first.id
    assert second.stop_loss == pytest.approx(109 * .9998)
    assert second.entry != first.entry


@pytest.mark.parametrize("wait,accepted", [(12, True), (13, False)])
def test_wait_window_is_measured_from_first_breach_and_does_not_restart(wait, accepted):
    frames = _frames()
    frames["5m"] = frames["5m"].iloc[:-1]
    rows = [(112, 119.5, 111.8, 112 + .5 * step) for step in range(1, wait + 1)]
    now = _append_entry(frames, rows)
    assert (_diagnose(frames, now=now).setup is not None) is accepted


def test_raid_identity_survives_a_new_liquidity_close_that_contains_the_sweep():
    frames = _frames()
    first = _diagnose(frames).setup
    assert first is not None
    rows = [(112, 119.5, 111.8, 117 + .1 * step) for step in range(1, 11)]
    now = _append_entry(frames, rows)
    consumed_bar = pd.DataFrame(
        [(118, 119.5, 109.5, 118)], columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex(["2026-10-05T07:00Z"]), dtype=float,
    )
    frames["1h"] = pd.concat([frames["1h"], consumed_bar])
    later = _diagnose(frames, now=now)
    assert later.setup is not None
    assert later.setup.id == first.id
    assert later.setup.stop_loss == first.stop_loss
    assert later.features["raid_start_open_utc"] == "2026-10-05T07:05:00+00:00"


def test_missing_entry_bar_at_the_last_known_liquidity_close_rejects():
    frames = _frames()
    assert _diagnose(frames).setup is not None
    frames["5m"] = frames["5m"].drop(pd.Timestamp("2026-10-05T07:00Z"))
    # An earlier bar cannot prove that the unobserved boundary bar did not raid.
    assert _diagnose(frames).setup is None


def test_gap_between_liquidity_close_and_first_breach_cannot_hide_an_earlier_raid():
    frames = _frames()
    entry = frames["5m"]
    tail = entry.iloc[-2:].copy()
    tail.index = tail.index + pd.Timedelta(minutes=5)
    pre_raid = pd.DataFrame(
        [(114, 115, 111, 112.5)], columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex(["2026-10-05T07:05Z"]), dtype=float,
    )
    frames["5m"] = pd.concat([entry.iloc[:-2], pre_raid, tail])
    now = datetime(2026, 10, 5, 7, 20, tzinfo=timezone.utc)
    complete = _diagnose(frames, now=now)
    assert complete.setup is not None
    assert complete.features["raid_start_open_utc"] == "2026-10-05T07:10:00+00:00"
    frames["5m"] = frames["5m"].drop(pd.Timestamp("2026-10-05T07:05Z"))
    assert _diagnose(frames, now=now).setup is None


def test_equal_low_plateau_is_causally_confirmed_then_collapsed_only_in_equal_mode():
    frames = _frames()
    # Preserve the original strict HH/HL bias; append a newer equal plateau.
    # The plateau consumes strict low110, while low105 is below the M5 raid.
    hourly = frames["1h"].copy()
    hourly.index = hourly.index - pd.Timedelta(hours=5)
    extra = pd.DataFrame(
        [(118, 124, 115, 119), (115, 124, 109, 115), (115, 124, 109, 116),
         (118, 124, 116, 120), (120, 124, 117, 121)],
        columns=["open", "high", "low", "close"], dtype=float,
        index=pd.date_range("2026-10-05 02:00", periods=5, freq="h", tz="UTC"),
    )
    frames["1h"] = pd.concat([hourly, extra])
    frames["5m"].iloc[-2, frames["5m"].columns.get_loc("low")] = 108.5
    assert v1._trend(v1._pivots(frames["1h"], 2)) == "LONG"
    assert _diagnose(frames, parameters=V5Parameters(liquidity_mode="strict")).setup is None
    assert _diagnose(frames, parameters=V5Parameters(liquidity_mode="equal_clusters")).setup is not None


def test_equal_plateau_requires_all_right_confirmation_candles_to_be_closed():
    from app.strategy.volium_v5 import _live_levels

    hourly = _frames()["1h"].copy()
    hourly.iloc[9] = [117, 120, 109, 119]
    hourly.iloc[10] = [117, 121, 109, 119]
    before_last_right_close = v1._closed(hourly, "1h", pd.Timestamp("2026-10-05T06:59Z"))
    after_last_right_close = v1._closed(hourly, "1h", pd.Timestamp("2026-10-05T07:00Z"))
    assert not any(level.price == 109 for level in _live_levels(before_last_right_close, 2, "equal_clusters", "low"))
    assert any(level.price == 109 for level in _live_levels(after_last_right_close, 2, "equal_clusters", "low"))


@pytest.mark.parametrize("short", [False, True])
def test_separated_equal_extremes_require_confirmation_after_last_touch(short):
    from app.strategy.volium_v5 import _live_levels

    hourly = _frame([120] * 8, [112, 111, 110, 111, 110, 110, 111, 112],
                    "2026-10-05 06:00", "h")
    if short:
        hourly = _mirror({"1h": hourly})["1h"]
    kind, price = ("high", 140) if short else ("low", 110)
    before = v1._closed(hourly, "1h", pd.Timestamp("2026-10-05T06:59Z"))
    closed = v1._closed(hourly, "1h", pd.Timestamp("2026-10-05T07:00Z"))
    assert not any(level.price == price for level in _live_levels(before, 2, "equal_clusters", kind))
    pool = [level for level in _live_levels(closed, 2, "equal_clusters", kind) if level.price == price]
    assert len(pool) == 1
    assert pool[0].index == 2
    assert pool[0].members == (2, 4, 5)
    assert pool[0].known_index == 7
    assert not any(level.price == price for level in _live_levels(closed, 2, "strict", kind))
    future = pd.DataFrame(
        [(115, 220, 50, 116)], columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex(["2026-10-05T07:00Z"]), dtype=float,
    )
    future_closed = v1._closed(pd.concat([hourly, future]), "1h", pd.Timestamp("2026-10-05T07:00Z"))
    assert _live_levels(future_closed, 2, "equal_clusters", kind) == _live_levels(closed, 2, "equal_clusters", kind)


@pytest.mark.parametrize("short", [False, True])
def test_separated_equal_pool_survives_equal_touch_but_actual_crossing_consumes(short):
    from app.strategy.volium_v5 import _live_levels

    hourly = _frame([120] * 11, [112, 111, 110, 111, 110, 110, 111, 112, 110, 111, 112],
                    "2026-10-05 09:00", "h")
    if short:
        hourly = _mirror({"1h": hourly})["1h"]
    kind, price = ("high", 140) if short else ("low", 110)
    assert any(level.price == price for level in _live_levels(hourly, 2, "equal_clusters", kind))
    crossed = hourly.copy()
    crossed.iloc[-1, crossed.columns.get_loc("high" if short else "low")] = 140.000001 if short else 109.999999
    assert not any(level.price == price for level in _live_levels(crossed, 2, "equal_clusters", kind))


def test_exact_equal_touch_preserves_equal_pool_but_consumes_strict_pool():
    frames = _frames()
    frames["1h"].iloc[-2] = [117, 122, 110, 119]
    assert _diagnose(frames, parameters=V5Parameters(liquidity_mode="strict")).setup is None
    assert _diagnose(frames, parameters=V5Parameters(liquidity_mode="equal_clusters")).setup is not None


@pytest.mark.parametrize("liquidity_mode", ["strict", "equal_clusters"])
def test_future_and_forming_candles_cannot_rewrite_any_diagnostic(liquidity_mode):
    frames = _frames()
    parameters = V5Parameters(liquidity_mode=liquidity_mode)
    reference = _diagnose(frames, parameters=parameters).as_dict()
    assert reference["accepted"]
    for timeframe, frame in list(frames.items()):
        extra = pd.DataFrame(
            [(160, 250, 50, 180)], columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex(["2026-10-06T00:00Z"]),
        )
        frames[timeframe] = pd.concat([frame, extra])
    forming = pd.DataFrame(
        [(117, 250, 50, 180)], columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex(["2026-10-05T07:15Z"]),
    )
    frames["5m"] = pd.concat([frames["5m"].iloc[:-1], forming, frames["5m"].iloc[-1:]])
    assert _diagnose(frames, parameters=parameters).as_dict() == reference


def test_consumed_daily_destination_and_consumed_liquidity_target_reject():
    daily_consumed = _frames()
    daily_consumed["1d"].iloc[-1] = [124, 145, 116, 141]
    assert _diagnose(daily_consumed).setup is None
    hourly_consumed = _frames()
    hourly_consumed["1h"].iloc[-1] = [120, 131, 117, 121]
    assert _diagnose(hourly_consumed).setup is None


def _scalp_frames():
    base = _frames()
    context = base["1d"].copy()
    context.index = pd.date_range("2026-10-04 13:00", periods=len(context), freq="h", tz="UTC")
    liquidity = base["1h"].copy()
    liquidity.index = pd.date_range("2026-10-05 06:00", periods=len(liquidity), freq="5min", tz="UTC")
    liquidity.iloc[11, liquidity.columns.get_loc("high")] = 125
    liquidity = pd.concat([liquidity, pd.DataFrame(
        [(120, 124, 117, 122)], columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex(["2026-10-05T07:05Z"]), dtype=float,
    )])
    entry = base["5m"].iloc[-4:].copy()
    entry.index = pd.date_range("2026-10-05 07:08", periods=len(entry), freq="min", tz="UTC")
    return {"1h": context, "5m": liquidity, "1m": entry}


@pytest.mark.parametrize("short", [False, True])
def test_declared_scalp_transfer_uses_h1_m5_m1_and_nearest_unhit_target(short):
    frames = _scalp_frames()
    settings = SimpleNamespace(volium_active_trend_min_atr=0)
    if short:
        frames = _mirror(frames)
    direction = "SHORT" if short else "LONG"
    assert v1._trend(v1._pivots(frames["1h"], 2)) == direction
    assert v1._active_trend(frames["1h"], direction, 0)
    result = _diagnose(frames, mode="scalp", settings=settings,
                       now=datetime(2026, 10, 5, 7, 12, tzinfo=timezone.utc))
    assert result.setup is not None
    assert result.setup.setup_type == "VOLIUM_SCALP"
    assert result.setup.take_profits == [125]
    assert result.setup.stop_loss == pytest.approx(140.5 * 1.001 if short else 109.5 * .999)
    assert result.features["target_known_close_utc"] <= result.features["raid_start_open_utc"]


def test_scalp_preserves_v1_active_trend_requirement():
    result = _diagnose(_scalp_frames(), mode="scalp",
                       settings=SimpleNamespace(volium_active_trend_min_atr=100),
                       now=datetime(2026, 10, 5, 7, 12, tzinfo=timezone.utc))
    assert result.setup is None
    assert result.reason == "inactive_hourly_trend"


def test_batch_and_factory_match_direct_immutable_parameter_variants():
    parameter_sets = [V5Parameters(liquidity_mode=mode) for mode in ("strict", "equal_clusters", "strict")]
    kwargs = dict(symbol="BTC_USDT", frames=_frames(), now=NOW, enforce_session_filter=False)
    direct = [diagnose_volium_v5_from_df(**kwargs, parameters=parameters) for parameters in parameter_sets]
    batched = diagnose_volium_v5_batch_from_df(**kwargs, parameter_sets=parameter_sets)
    assert [result.as_dict() for result in batched] == [result.as_dict() for result in direct]
    assert [setup.model_dump() if setup else None for setup in analyze_volium_v5_batch_from_df(**kwargs, parameter_sets=parameter_sets)] == [result.setup.model_dump() if result.setup else None for result in direct]
    assert make_v5_analyzer(parameter_sets[0])(**kwargs) == analyze_volium_v5_from_df(**kwargs, parameters=parameter_sets[0])


@pytest.mark.parametrize("kwargs", [
    {"liquidity_mode": "epsilon"}, {"recovery_fraction": .25},
    {"recovery_fraction": float("nan")}, {"max_wait_bars": 13},
    {"max_wait_bars": 0}, {"max_wait_bars": 12.5}, {"max_wait_bars": 12.0},
])
def test_numeric_hypothesis_and_supported_liquidity_variants_are_fixed(kwargs):
    with pytest.raises(ValueError):
        V5Parameters(**kwargs)


def test_parameters_are_immutable():
    parameters = V5Parameters()
    with pytest.raises(FrozenInstanceError):
        parameters.liquidity_mode = "equal_clusters"


def test_invalid_mode_and_missing_or_corrupt_inputs_fail_explicitly():
    with pytest.raises(ValueError):
        _diagnose(mode="swing")
    assert _diagnose({"1d": _frames()["1d"]}).setup is None
    frames = _frames()
    frames["5m"].iloc[-1, frames["5m"].columns.get_loc("low")] = float("nan")
    assert _diagnose(frames).setup is None
