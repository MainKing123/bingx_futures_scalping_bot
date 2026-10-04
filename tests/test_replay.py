from datetime import datetime, timezone

import pandas as pd
import pytest

from app.config import Settings
from app.replay import SECONDS, replay, required_timeframes
from app.schemas.setup import TradeSetup


def _settings(**overrides):
    values = dict(_env_file=None, mexc_api_key="", mexc_api_secret="", auto_execution=False,
                  volium_session_enabled=False, paper_fee_bps=0, paper_slippage_bps=0,
                  account_balance_usdt=1000, daily_loss_limit_percent=20)
    values.update(overrides)
    return Settings(**values)


def _frame(rows, tf="5m", start="2026-10-05 07:00", index=None):
    if index is None:
        index = pd.date_range(start, periods=len(rows), freq=f"{SECONDS[tf]}s", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index, dtype=float)


def _frames(entry, settings):
    result = {}
    for tf in required_timeframes(settings):
        if tf == required_timeframes(settings)[-1]:
            result[tf] = entry
        else:
            idx = pd.date_range(end=entry.index[0], periods=150, freq=f"{SECONDS[tf]}s", tz="UTC")
            result[tf] = pd.DataFrame({"open":105., "high":106., "low":104., "close":105.}, index=idx)
    return result


def _setup(now, identifier="idea", direction="LONG", mode="intraday"):
    return TradeSetup(id=identifier, timestamp=now, symbol="BTC_USDT", direction=direction,
                      setup_type=f"VOLIUM_{mode.upper()}", htf_bias="BULLISH" if direction == "LONG" else "BEARISH",
                      entry=100, stop_loss=90 if direction == "LONG" else 110,
                      take_profits=[120 if direction == "LONG" else 80], risk_reward=2, confluences=[])


def _first_only(direction="LONG"):
    emitted = False

    def provider(**kwargs):
        nonlocal emitted
        if emitted:
            return None
        emitted = True
        return _setup(kwargs["now"], direction=direction, mode=kwargs["mode"])
    return provider


def _run(rows, *, settings=None, direction="LONG", **kwargs):
    settings = settings or _settings()
    tf = required_timeframes(settings)[-1]
    entry = _frame(rows, tf=tf)
    return replay("BTC_USDT", _frames(entry, settings), settings, signal_provider=_first_only(direction), **kwargs)


def test_pending_limit_must_fill_before_any_profit_is_counted():
    result = _run([(105,106,104,105), (105,125,101,110), (105,107,99,102), (110,121,109,120)])
    assert result["signals"] == result["trades_count"] == 1
    assert result["trades"][0]["entry_time"] == "2026-10-05T07:10:00+00:00"
    assert result["trades"][0]["exit_time"] == "2026-10-05T07:20:00+00:00"
    assert result["net_pnl_usdt"] == pytest.approx(10)


def test_target_touch_without_limit_touch_is_unfilled_not_profitable():
    result = _run([(105,106,104,105), (105,125,101,110)])
    assert result["trades_count"] == 0
    assert result["unfinished_limit"] is True
    assert result["unfinished_position"] is False
    assert result["unfilled_signals"] == 1
    assert result["net_pnl_usdt"] == 0


@pytest.mark.parametrize("direction,expected_exit", [("LONG",90), ("SHORT",110)])
def test_stop_has_priority_when_both_stop_and_target_are_touched(direction, expected_exit):
    result = _run([(105,106,104,105), (100,125,75,100)], direction=direction)
    trade = result["trades"][0]
    assert trade["status"] == "SL_HIT"
    assert trade["exit"] == expected_exit
    assert trade["pnl_usdt"] == pytest.approx(-5)


def test_uncertain_fill_bar_target_is_deferred_until_later_bar():
    result = _run([(105,106,104,105), (105,125,99,110), (110,121,109,120)])
    assert result["trades"][0]["exit_time"] == "2026-10-05T07:15:00+00:00"
    assert result["trades"][0]["status"] == "TP1_HIT"


def test_open_through_limit_has_known_fill_before_same_bar_target():
    result = _run([(105,106,104,105), (99,121,95,115)])
    assert result["trades_count"] == 1
    assert result["trades"][0]["exit_time"] == "2026-10-05T07:10:00+00:00"


def test_gap_beyond_stop_records_adverse_open_not_artificial_stop_price():
    result = _run([(105,106,104,105), (100,105,95,100), (85,89,84,87)])
    assert result["trades"][0]["exit"] == 85
    assert result["net_pnl_usdt"] == pytest.approx(-7.5)


def test_round_trip_fees_and_slippage_reduce_profit():
    result = _run([(105,106,104,105), (100,121,95,115)], settings=_settings(paper_fee_bps=5, paper_slippage_bps=2))
    assert result["trades"][0]["notional"] == pytest.approx(50)
    assert result["net_pnl_usdt"] == pytest.approx(50 * (.2 - .0014))
    assert result["final_realized_equity"] == pytest.approx(1009.93)


def test_sizing_compounds_realized_equity_and_locks_notional_for_each_trade():
    settings = _settings()
    df = _frame([(105,106,104,105), (100,121,95,115), (100,121,95,115)])
    calls = 0

    def provider(**kwargs):
        nonlocal calls
        calls += 1
        return _setup(kwargs["now"], identifier=f"idea{calls}") if calls <= 2 else None

    result = replay("BTC_USDT", _frames(df, settings), settings, signal_provider=provider)
    assert [t["notional"] for t in result["trades"]] == pytest.approx([50,50.5])
    assert result["final_realized_equity"] == pytest.approx(1020.1)


def test_duplicate_sweep_id_cannot_be_traded_again_after_exit():
    settings = _settings()
    df = _frame([(105,106,104,105), (100,121,95,115), (100,121,95,115)])
    result = replay("BTC_USDT", _frames(df, settings), settings,
                    signal_provider=lambda **kw: _setup(kw["now"], identifier="same-sweep"))
    assert result["signals"] == result["trades_count"] == 1


def test_expiration_applies_only_to_unfilled_limit_and_excludes_exact_expiry_open():
    result = _run([(105,106,104,105), (105,110,101,105), (100,121,95,115)],
                  settings=_settings(pending_order_max_age_minutes=5))
    assert result["trades_count"] == 0
    assert result["expired_limits"] == 1
    assert result["unfinished_limit"] is False


def test_swing_position_holds_after_pending_ttl_when_already_filled():
    result = _run([(105,106,104,105), (100,105,95,101), (101,110,99,105), (110,121,109,120)],
                  settings=_settings(volium_mode="swing", pending_order_max_age_minutes=1))
    assert result["trades_count"] == 1
    assert result["expired_limits"] == 0
    assert result["trades"][0]["entry_time"] == "2026-10-05T08:00:00+00:00"
    assert result["trades"][0]["exit_time"] == "2026-10-05T11:00:00+00:00"


def test_swing_partial_bar_expiry_does_not_assume_ambiguous_fill_succeeded():
    result = _run([(105,106,104,105), (105,121,99,110)],
                  settings=_settings(volium_mode="swing", pending_order_max_age_minutes=30))
    assert result["trades_count"] == 0
    assert result["expired_limits"] == result["ambiguous_expiry_limits"] == 1
    assert result["net_pnl_usdt"] == 0


def test_one_minute_execution_can_resolve_swing_fill_before_expiry_causally():
    settings = _settings(volium_mode="swing", pending_order_max_age_minutes=30)
    entry = _frame([(105,106,104,105), (105,121,99,120)], tf="1h")
    execution = _frame([(105,106,104,105)] * 61, tf="1m", start="2026-10-05 07:59")
    execution.loc[pd.Timestamp("2026-10-05T08:10:00Z")] = [105,106,99,101]
    execution.loc[pd.Timestamp("2026-10-05T08:40:00Z")] = [110,121,109,120]
    calls = []
    emitted = False

    def provider(**kwargs):
        nonlocal emitted
        calls.append(kwargs["now"])
        if emitted:
            return None
        emitted = True
        return _setup(kwargs["now"], mode="swing")

    result = replay("BTC_USDT", _frames(entry, settings), settings, execution_frame=execution, signal_provider=provider)
    assert result["execution_timeframe"] == "1m"
    assert result["trades_count"] == 1
    assert result["trades"][0]["entry_time"] == "2026-10-05T08:10:00+00:00"
    assert result["trades"][0]["exit_time"] == "2026-10-05T08:41:00+00:00"
    assert calls == [datetime(2026,10,5,8,tzinfo=timezone.utc), datetime(2026,10,5,9,tzinfo=timezone.utc)]


def test_future_higher_timeframe_candles_never_reach_signal_provider_even_if_unsorted():
    settings = _settings()
    entry = _frame([(105,106,104,105)] * 12)
    frames = _frames(entry, settings)
    future = _frame([(1000,2000,500,1500)], tf="1h", start="2026-10-06 00:00")
    frames["1h"] = pd.concat([future, frames["1h"].iloc[::-1]])
    observations = []

    def provider(**kwargs):
        moment = pd.Timestamp(kwargs["now"])
        assert set(required_timeframes(settings)).issubset(kwargs["frames"])
        for tf, frame in kwargs["frames"].items():
            assert (frame.index + pd.Timedelta(seconds=SECONDS[tf]) <= moment).all()
            assert len(frame) <= settings.volium_context_lookback + 30
        observations.append(moment)
        return None

    result = replay("BTC_USDT", frames, settings, signal_provider=provider)
    assert len(observations) == len(entry)
    assert result["signals"] == 0


def test_daily_loss_limit_resets_on_utc_calendar_date():
    settings = _settings(daily_loss_limit_percent=.5)
    idx = pd.DatetimeIndex(["2026-10-05T23:35:00Z","2026-10-05T23:40:00Z", "2026-10-05T23:45:00Z", "2026-10-06T00:00:00Z"])
    entry = _frame([(105,106,104,105), (100,101,89,90), (105,106,104,105), (105,106,104,105)], index=idx)
    observed = []

    def provider(**kwargs):
        observed.append(kwargs["now"])
        return _setup(kwargs["now"]) if len(observed) == 1 else None

    result = replay("BTC_USDT", _frames(entry, settings), settings, signal_provider=provider)
    assert result["net_pnl_usdt"] == pytest.approx(-5)
    assert observed == [datetime(2026,10,5,23,40,tzinfo=timezone.utc), datetime(2026,10,6,0,5,tzinfo=timezone.utc)]


def test_daily_loss_threshold_is_frozen_to_day_opening_equity():
    settings = _settings(risk_per_trade_percent=1, daily_loss_limit_percent=2)
    entry = _frame([(105,106,104,105)] + [(100,101,89,90)] * 4)
    count = 0

    def provider(**kwargs):
        nonlocal count
        count += 1
        return _setup(kwargs["now"], identifier=f"loss{count}")

    result = replay("BTC_USDT", _frames(entry, settings), settings, signal_provider=provider)
    # The first two compounded losses total19.9, still below the fixed20 threshold.
    assert result["signals"] == result["trades_count"] == 3
    assert result["net_pnl_usdt"] == pytest.approx(-29.701)


@pytest.mark.parametrize("mode,context,expected", [
    ("intraday","1d",("1d","1h","5m")), ("scalp","1d",("1h","5m","1m")),
    ("swing","1d",("1d","1h")), ("swing","1w",("1w","4h")),
])
def test_modes_request_their_own_context_and_entry_frames(mode, context, expected):
    assert required_timeframes(_settings(volium_mode=mode, volium_swing_context=context)) == expected


def test_missing_context_frames_are_rejected_instead_of_reporting_empty_success():
    with pytest.raises(ValueError, match="Missing strategy frames"):
        replay("BTC_USDT", {"5m":_frame([(105,106,104,105)])}, _settings())


def _with_funding(rows, rates, direction="LONG", **settings_overrides):
    settings = _settings(**settings_overrides)
    entry = _frame(rows)
    return replay("BTC_USDT", _frames(entry, settings), settings,
                  signal_provider=_first_only(direction), funding_rates=rates)


def test_long_pays_settled_positive_funding_and_marked_equity_tracks_open_position():
    rates = pd.Series([.001,.002], index=pd.DatetimeIndex(["2026-10-05T07:10:00Z","2026-10-05T07:15:00Z"]))
    result = _with_funding([(105,106,104,105), (100,110,95,105), (105,111,104,110)], rates)
    assert result["funding_total_usdt"] == pytest.approx(-.5*105*.001-.5*110*.002)
    assert result["funding_events_charged"] == 2
    assert result["trades_count"] == 0 and result["unfinished_position"] is True
    assert result["unrealized_pnl_usdt"] == pytest.approx(5)
    assert result["marked_equity"] == pytest.approx(result["final_realized_equity"]+5)


def test_short_receives_positive_funding_when_known_held_across_settlement():
    rates = pd.Series([.001], index=pd.DatetimeIndex(["2026-10-05T07:10:00Z"]))
    result = _with_funding([(105,106,104,105), (100,105,90,95), (95,96,85,90)], rates, direction="SHORT")
    assert result["funding_total_usdt"] == pytest.approx(.5*95*.001)
    assert result["funding_events_skipped_favorable"] == 0
    assert result["unrealized_pnl_usdt"] == pytest.approx(5)


def test_duplicate_settlements_are_not_charged_again_and_future_rates_are_ignored():
    rates = pd.Series([.001,.002,1.0], index=pd.DatetimeIndex([
        "2026-10-05T07:10:00Z","2026-10-05T07:10:00Z","2026-10-06T00:00:00Z"]))
    result = _with_funding([(105,106,104,105), (100,110,95,105), (105,111,104,110)], rates)
    assert result["funding_total_usdt"] == pytest.approx(-.5*105*.002)
    assert result["funding_events_charged"] == 1


@pytest.mark.parametrize("direction,rate,expected", [("LONG",-.001,0), ("LONG",.001,-.0525),
                                                     ("SHORT",.001,0), ("SHORT",-.001,-.0475)])
def test_ambiguous_entry_bar_cannot_gain_speculative_favorable_funding(direction, rate, expected):
    rates = pd.Series([rate], index=pd.DatetimeIndex(["2026-10-05T07:10:00Z"]))
    row = (105,110,99,105) if direction == "LONG" else (95,101,90,95)
    result = _with_funding([(105,106,104,105), row], rates, direction=direction)
    assert result["funding_total_usdt"] == pytest.approx(expected)
    assert result["funding_events_skipped_favorable"] == int(expected == 0)


def test_closed_trade_funding_is_in_trade_pnl_and_equity_exactly_once():
    rates = pd.Series([.001], index=pd.DatetimeIndex(["2026-10-05T07:10:00Z"]))
    result = _with_funding([(105,106,104,105), (100,110,95,105), (110,121,109,120)], rates,
                           paper_fee_bps=5,paper_slippage_bps=2)
    trade = result["trades"][0]
    assert trade["price_pnl_usdt"] == pytest.approx(9.93)
    assert trade["funding_usdt"] == pytest.approx(-.0525)
    assert trade["pnl_usdt"] == pytest.approx(9.8775)
    assert result["final_realized_equity"] == pytest.approx(1009.8775)
    assert result["unrealized_pnl_usdt"] == 0


def test_known_exit_at_bar_open_cannot_be_charged_later_funding():
    rates = pd.Series([.001], index=pd.DatetimeIndex(["2026-10-05T07:15:00Z"]))
    result = _with_funding([(105,106,104,105), (100,105,95,100), (125,126,124,125)], rates)
    assert result["funding_total_usdt"] == 0
    assert result["trades"][0]["funding_usdt"] == 0
    assert result["net_pnl_usdt"] == pytest.approx(10)


def test_marked_drawdown_reports_open_loss_before_stop_or_realized_exit():
    result = _run([(105,106,104,105), (100,106,95,105), (105,106,94,95)])
    assert result["max_realized_drawdown_percent"] == 0
    assert result["unrealized_pnl_usdt"] == pytest.approx(-2.5)
    assert result["marked_equity"] == pytest.approx(997.5)
    assert result["max_marked_drawdown_percent"] == pytest.approx((1002.5-997.5)/1002.5*100)
