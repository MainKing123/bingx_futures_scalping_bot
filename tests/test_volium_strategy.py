from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from app.strategy.volium import analyze_volium_from_df, in_volium_session, is_volium_session, rr2_entry


NOW = datetime(2026, 10, 5, 7, 15, tzinfo=timezone.utc)


def _frame(highs, lows, end, freq):
    idx = pd.date_range(end=end, periods=len(highs), freq=freq, tz="UTC")
    middle = [(hi + lo) / 2 for hi, lo in zip(highs, lows)]
    return pd.DataFrame({"open": middle, "high": highs, "low": lows, "close": middle}, index=idx)


def _fixtures():
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
    entry = pd.DataFrame(
        [(117, 119, 116, 118), (118, 119, 113, 114), (114, 115, 109.5, 112), (112, 117.2, 111.8, 117)],
        columns=["open", "high", "low", "close"], dtype=float,
        index=pd.date_range("2026-10-05 06:55", periods=4, freq="5min", tz="UTC"),
    )
    return {"1d": daily, "1h": hourly, "5m": entry}


def _signal(frames=None, **kwargs):
    return analyze_volium_from_df(symbol="BTC_USDT", frames=frames or _fixtures(), now=NOW, **kwargs)


def test_intraday_complete_checklist_fixed_structural_target_exact_2r():
    setup = _signal()
    assert setup is not None
    assert setup.setup_type == "VOLIUM_INTRADAY"
    assert setup.direction == "LONG"
    assert setup.take_profits == [130]
    assert setup.stop_loss < 109.5
    assert setup.entry < 117  # A pending retest limit, not the confirmation price.
    assert (130 - setup.entry) / (setup.entry - setup.stop_loss) == pytest.approx(2)
    assert setup.timestamp == NOW
    assert setup.id == _signal().id


def test_short_is_symmetric_and_risk_geometry_valid():
    frames = _fixtures()
    for key, df in frames.items():
        mirrored = df.copy()
        mirrored["open"] = 250 - df.open
        mirrored["close"] = 250 - df.close
        mirrored["high"] = 250 - df.low
        mirrored["low"] = 250 - df.high
        frames[key] = mirrored
    setup = _signal(frames)
    assert setup is not None
    assert setup.direction == "SHORT"
    assert setup.take_profits == [120]
    assert 120 < setup.entry < setup.stop_loss
    assert (setup.entry - 120) / (setup.stop_loss - setup.entry) == pytest.approx(2)


def test_same_sweep_cannot_generate_distinct_ideas_on_later_confirmation():
    frames = _fixtures()
    first = _signal(frames)
    following = pd.DataFrame([[117, 119.2, 116.9, 119]], columns=["open", "high", "low", "close"],
                              index=pd.DatetimeIndex(["2026-10-05T07:15:00Z"]))
    frames["5m"] = pd.concat([frames["5m"], following])
    second = analyze_volium_from_df(symbol="BTC_USDT", frames=frames,
                                    now=datetime(2026, 10, 5, 7, 20, tzinfo=timezone.utc))
    assert first is not None and second is not None
    assert first.id == second.id
    assert first.entry == second.entry


def test_sweep_alone_without_body_clearance_is_not_a_signal():
    frames = _fixtures()
    frames["5m"].iloc[-1] = [112, 114.5, 111.8, 113]
    assert _signal(frames) is None


def test_weak_choppy_reaction_is_not_a_signal():
    frames = _fixtures()
    frames["5m"].iloc[-1] = [116.7, 117.2, 111.8, 117]
    assert _signal(frames) is None


def test_unclosed_entry_candle_cannot_confirm_signal():
    frames = _fixtures()
    assert analyze_volium_from_df(symbol="BTC_USDT", frames=frames, now=datetime(2026, 10, 5, 7, 14, tzinfo=timezone.utc)) is None


def test_future_context_and_entry_candles_do_not_change_signal():
    frames = _fixtures()
    reference = _signal(frames)
    for key, df in list(frames.items()):
        extra = pd.DataFrame([[200, 210, 190, 205]], columns=["open", "high", "low", "close"],
                             index=pd.DatetimeIndex(["2026-10-06T00:00:00Z"]))
        frames[key] = pd.concat([df, extra])
    result = _signal(frames)
    assert result is not None and reference is not None
    assert result.model_dump() == reference.model_dump()


def test_context_caches_invalidate_when_same_frame_values_change():
    frames = _fixtures()
    original = _signal(frames)
    assert original is not None
    # Reusing a DataFrame object must not reuse the old structural target.
    frames["1h"].iloc[8, frames["1h"].columns.get_loc("high")] = 131
    changed = _signal(frames)
    assert changed is not None
    assert changed.take_profits == [131]
    assert changed.entry != original.entry
    # Consuming the daily point A without a strict sweep removes its reclaim.
    frames["1d"].iloc[6, frames["1d"].columns.get_loc("low")] = 80
    assert _signal(frames) is None


def test_cached_pivot_results_cannot_be_mutated_by_a_caller():
    from app.strategy.volium import _pivots

    frame = _fixtures()["1d"]
    expected = _pivots(frame, 2)
    caller_list = _pivots(frame, 2)
    caller_list.clear()
    assert _pivots(frame, 2) == expected


def test_target_already_touched_or_no_remaining_daily_destination_blocks_trade():
    frames = _fixtures()
    frames["1d"].iloc[-1] = [124, 145, 116, 141]
    assert _signal(frames) is None


def test_daily_origin_is_required_but_numeric_interpretation_is_configurable():
    frames = _fixtures()
    frames["1d"].iloc[6, frames["1d"].columns.get_loc("low")] = 82
    assert _signal(frames) is None
    assert _signal(frames, settings=SimpleNamespace(volium_require_daily_origin_sweep=False)) is not None


def test_invalid_ohlc_missing_frames_and_stale_input_fail_closed():
    frames = _fixtures()
    frames["5m"].iloc[-1, frames["5m"].columns.get_loc("high")] = float("nan")
    assert _signal(frames) is None
    assert analyze_volium_from_df(symbol="BTC_USDT", frames={"5m": _fixtures()["5m"]}, now=NOW) is None
    assert analyze_volium_from_df(symbol="BTC_USDT", frames=_fixtures(), now=datetime(2026, 10, 5, 7, 30, tzinfo=timezone.utc)) is None


def test_session_uses_fixed_utc3_and_exclusive_end():
    windows = [("10:00", "12:00")]
    assert not is_volium_session(datetime(2026, 10, 5, 6, 59, tzinfo=timezone.utc), windows)
    assert is_volium_session(datetime(2026, 10, 5, 7, 0, tzinfo=timezone.utc), windows)
    assert is_volium_session(datetime(2026, 10, 5, 8, 59, tzinfo=timezone.utc), windows)
    assert not is_volium_session(datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc), windows)
    assert not is_volium_session(datetime(2026, 10, 5, 13, 30, tzinfo=timezone.utc), windows)


def test_explicit_afternoon_interpretation_and_morning_only_sensitivity():
    frames = _fixtures()
    frames["5m"].index += pd.Timedelta(hours=6, minutes=30)
    frames["1h"].index += pd.Timedelta(hours=6)
    afternoon = datetime(2026, 10, 5, 13, 45, tzinfo=timezone.utc)
    assert analyze_volium_from_df(symbol="BTC_USDT", frames=frames, now=afternoon) is not None
    assert analyze_volium_from_df(symbol="BTC_USDT", frames=frames, now=afternoon,
                                  settings=SimpleNamespace(volium_sessions_utc3=[("10:00", "12:00")])) is None


@pytest.mark.parametrize("zone,start,end,dates,start_utc", [
    ("America/New_York","09:30","11:00", ["2026-03-06","2026-03-09","2026-11-02"], ["14:30","13:30","14:30"]),
    ("Europe/London","08:00","10:00", ["2026-03-09","2026-03-30","2026-11-02"], ["08:00","07:00","08:00"]),
])
def test_market_local_sessions_follow_independent_us_uk_dst_dates(zone, start, end, dates, start_utc):
    settings = SimpleNamespace(volium_session_clock="market_local", volium_market_sessions=[(zone,start,end)])
    for date, utc_start in zip(dates, start_utc):
        timestamp = pd.Timestamp(f"{date}T{utc_start}:00Z")
        assert in_volium_session(timestamp, settings)
        assert not in_volium_session(timestamp - pd.Timedelta(minutes=1), settings)
        length = (int(end[:2])*60+int(end[3:])) - (int(start[:2])*60+int(start[3:]))
        assert not in_volium_session(timestamp + pd.Timedelta(minutes=length), settings)


def test_default_fixed_utc3_table_does_not_shift_in_winter():
    assert in_volium_session(pd.Timestamp("2026-11-02T07:00:00Z"))
    assert in_volium_session(pd.Timestamp("2026-11-02T13:30:00Z"))


def test_swing_does_not_require_intraday_sessions():
    frames = _fixtures()
    # D1 has an unswept low105; the H1 response takes it and clears the bearish body.
    context = frames["1d"]
    hourly = pd.DataFrame([(116, 119, 111, 118), (118, 119, 108, 110), (110, 112, 104, 106), (106, 119.5, 105.5, 119)],
                         columns=["open", "high", "low", "close"],
                         index=pd.date_range("2026-10-05 10:00", periods=4, freq="h", tz="UTC"))
    setup = analyze_volium_from_df(symbol="BTC_USDT", frames={"1d": context, "1h": hourly}, mode="swing",
                                   now=datetime(2026, 10, 5, 14, 0, tzinfo=timezone.utc))
    assert setup is not None
    assert setup.setup_type == "VOLIUM_SWING"
    assert setup.take_profits == [140]
    assert any("No session filter" in c for c in setup.confluences)


def test_scalp_uses_5m_destination_and_buffered_stop():
    frames = _fixtures()
    # Hourly trend must also exhibit at least one ATR of recent forward progress.
    hourly = frames["1h"].copy()
    hourly.iloc[-3] = [115, 122, 114, 120]
    hourly.iloc[-2] = [120, 124, 118, 123]
    hourly.iloc[-1] = [123, 129, 122, 128]
    five = frames["1h"].copy()
    five.index = pd.date_range("2026-10-05 06:00", periods=len(five), freq="5min", tz="UTC")
    minute = frames["5m"].copy()
    minute.index = pd.date_range("2026-10-05 07:11", periods=4, freq="min", tz="UTC")
    setup = analyze_volium_from_df(symbol="BTC_USDT", frames={"1h": hourly, "5m": five, "1m": minute},
                                   settings=SimpleNamespace(volium_active_trend_min_atr=0.5), mode="scalp", now=NOW)
    assert setup is not None
    assert setup.setup_type == "VOLIUM_SCALP"
    assert setup.take_profits == [130]
    assert setup.stop_loss == pytest.approx(109.5 * 0.999)


def test_weekly_swing_uses_closed_four_hour_response():
    context = _fixtures()["1d"].copy()
    context.index = pd.date_range(end="2026-09-28", periods=len(context), freq="7D", tz="UTC")
    four_hour = pd.DataFrame([(116, 119, 111, 118), (118, 119, 108, 110), (110, 112, 104, 106), (106, 119.5, 105.5, 119)],
                            columns=["open", "high", "low", "close"],
                            index=pd.date_range("2026-10-04 20:00", periods=4, freq="4h", tz="UTC"))
    settings = SimpleNamespace(volium_swing_context="1w")
    assert analyze_volium_from_df(symbol="BTC_USDT", frames={"1w": context, "4h": four_hour}, mode="swing",
                                  settings=settings, now=datetime(2026, 10, 5, 11, 59, tzinfo=timezone.utc)) is None
    setup = analyze_volium_from_df(symbol="BTC_USDT", frames={"1w": context, "4h": four_hour}, mode="swing",
                                   settings=settings, now=datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc))
    assert setup is not None
    assert setup.setup_type == "VOLIUM_SWING"
    assert setup.take_profits == [140]


def test_rr2_formula_and_invalid_risk_geometry():
    assert rr2_entry(90, 120) == 100
    assert rr2_entry(120, 90) == 110
    with pytest.raises(ValueError):
        rr2_entry(100, 100)
    with pytest.raises(ValueError):
        rr2_entry(float("nan"), 100)
