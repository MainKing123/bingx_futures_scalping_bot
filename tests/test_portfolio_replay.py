import pandas as pd
import pytest

from app.config import Settings
from app.portfolio_replay import portfolio_replay
from app.replay import SECONDS, replay, required_timeframes
from app.schemas.setup import TradeSetup


def _settings(**overrides):
    values = dict(_env_file=None, mexc_api_key="", mexc_api_secret="", auto_execution=False,
        account_balance_usdt=1000, risk_per_trade_percent=.5, default_leverage=3, max_open_setups=3,
        daily_loss_limit_percent=2, paper_fee_bps=0, paper_slippage_bps=0,
        volium_session_enabled=False, volium_mode="scalp")
    values.update(overrides)
    return Settings(**values)


def _frames(rows, settings, symbols=("BTC_USDT",), index=None):
    index = pd.date_range("2026-10-05 07:00", periods=len(rows), freq="min", tz="UTC") if index is None else index
    entry = pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close"], dtype=float)
    result = {}
    for symbol in symbols:
        result[symbol] = {"1m":entry.copy()}
        for tf in required_timeframes(settings):
            if tf == "1m":
                continue
            idx = pd.date_range(end=entry.index[0]+pd.Timedelta(days=1), periods=150, freq=f"{SECONDS[tf]}s")
            result[symbol][tf] = pd.DataFrame({"open":105.,"high":106.,"low":104.,"close":105.},index=idx)
    return result


def _setup(kwargs, *, identifier="idea", stop=90, target=120, direction="LONG"):
    return TradeSetup(id=identifier,timestamp=kwargs["now"],symbol=kwargs["symbol"],direction=direction,
        setup_type=f"VOLIUM_{kwargs['mode'].upper()}",htf_bias="BULLISH" if direction=="LONG" else "BEARISH",
        entry=100,stop_loss=stop,take_profits=[target],risk_reward=2,confluences=[])


def _first_each(**setup_kwargs):
    seen = set()
    def provider(**kwargs):
        if kwargs["symbol"] in seen:
            return None
        seen.add(kwargs["symbol"])
        return _setup(kwargs,**setup_kwargs)
    return provider


def _contract(**overrides):
    contract = {"contractSize": 1, "volUnit": 1, "minVol": 1, "maxVol": 850,
                "limitMaxVol": 850, "priceUnit": .01}
    contract.update(overrides)
    return contract


def test_positive_signal_schedule_preserves_full_execution_and_financial_results():
    settings = _settings()
    frames = _frames([(105,106,104,105),(100,105,99,102),(110,121,109,120),(120,121,119,120)], settings)
    timestamp = frames["BTC_USDT"]["1m"].index[0]+pd.Timedelta(minutes=1)
    calls = []
    def provider(**kwargs):
        calls.append(kwargs["now"])
        return _setup(kwargs) if kwargs["now"] == timestamp else None
    plain = portfolio_replay(frames, settings, signal_provider=provider)
    calls.clear()
    scheduled = portfolio_replay(frames, settings, signal_provider=provider, signal_schedule={"BTC_USDT":[timestamp]})
    for key in ("net_pnl_usdt","marked_equity","max_marked_drawdown_percent","funding_total_usdt",
                "execution_bars","execution_minutes","trades","daily_close","unfinished_positions"):
        assert scheduled[key] == plain[key]
    assert calls == [timestamp]
    assert scheduled["positive_signal_schedule_applied"] is True


@pytest.mark.parametrize("bad_schedule", [{}, {"BTC_USDT":["2026-10-05T07:00:30Z"]},
                                         {"BTC_USDT":["2026-10-05T07:01Z","2026-10-05T07:01Z"]}])
def test_positive_signal_schedule_rejects_missing_symbol_nonclose_and_duplicates(bad_schedule):
    settings = _settings()
    frames = _frames([(105,106,104,105)]*2,settings)
    with pytest.raises(ValueError, match="schedule"):
        portfolio_replay(frames,settings,signal_schedule=bad_schedule)


def test_contract_maximum_and_lot_rounding_reserve_only_actual_margin():
    settings = _settings()
    frames = _frames([(105,106,104,105)]*2,settings)
    result = portfolio_replay(frames,settings,signal_provider=_first_each(stop=99.99,target=100.02),
                              position_limits={"BTC_USDT":_contract(maxVol=7,limitMaxVol=7)})
    order = result["active_orders_positions"][0]
    assert order["notional"] == pytest.approx(700)
    assert result["peak_reserved_margin_usdt"] == pytest.approx(700/3)
    assert result["downsized_for_contract_limits"] == 1
    assert result["historical_contract_constraints_known"] is False


def test_contract_minimum_rejects_order_instead_of_increasing_risk():
    settings = _settings()
    frames = _frames([(105,106,104,105)]*2,settings)
    result = portfolio_replay(frames,settings,signal_provider=_first_each(),
                              position_limits={"BTC_USDT":_contract()})
    # Desired $50 notional cannot buy a whole contract at $100.
    assert result["accepted_signals"] == 0
    assert result["rejected_contract_limits"] == 1
    assert result["peak_reserved_margin_usdt"] == 0


def test_tick_rounding_keeps_sweep_stop_and_minimum_2r_cash_risk():
    settings = _settings()
    frames = _frames([(105,106,104,105),(100,100.1,99.7,99.9)],settings)
    def provider(**kwargs):
        return _setup(kwargs,stop=99.805,target=100.39).model_copy(update={"entry":100.})
    result = portfolio_replay(frames,settings,signal_provider=provider,
                              position_limits={"BTC_USDT":_contract(contractSize=.01,priceUnit=.1)})
    trade = result["trades"][0]
    assert result["rounded_contract_prices"] >= 1
    assert trade["entry"] == pytest.approx(99.9)
    assert trade["stop"] == pytest.approx(99.8)
    assert trade["target"] == pytest.approx(100.3)
    assert trade["stop"] < 99.805
    assert (trade["target"]-trade["entry"])/(trade["entry"]-trade["stop"]) >= 2
    assert trade["notional"] == pytest.approx(850 * .01 * 99.9)
    assert trade["pnl_usdt"] == pytest.approx(-.85)
    assert abs(trade["pnl_usdt"]) <= 5


def test_three_shared_slots_include_pending_orders_and_use_universe_order():
    settings = _settings()
    symbols = ["ZEC_USDT","SOL_USDT","DOGE_USDT","XRP_USDT","ETH_USDT"]
    frames = _frames([(105,106,104,105)]*2,settings,symbols)
    result = portfolio_replay(frames,settings,symbols=symbols,signal_provider=_first_each())
    assert result["signals"] == 5 and result["accepted_signals"] == 3
    assert result["rejected_slots"] == 2
    assert result["peak_active_orders_positions"] == result["unfinished_limits"] == 3
    assert [order["symbol"] for order in result["active_orders_positions"]] == symbols[:3]
    assert result["peak_reserved_margin_usdt"] == pytest.approx(3*50/3)


def test_pending_margin_cannot_be_borrowed_twice_and_later_risk_size_is_capped():
    settings = _settings()
    symbols = ["A_USDT","B_USDT","C_USDT"]
    frames = _frames([(105,106,104,105)]*2,settings,symbols)
    result = portfolio_replay(frames,settings,signal_provider=_first_each(stop=99.5,target=101))
    assert result["accepted_signals"] == 3
    assert result["downsized_for_margin"] == 1
    assert [order["notional"] for order in result["active_orders_positions"]] == pytest.approx([1000,1000,700])
    assert result["peak_reserved_margin_usdt"] == pytest.approx(900)
    assert result["peak_margin_utilization_percent"] == pytest.approx(90)


def test_no_remaining_shared_margin_rejects_other_pending_orders():
    settings = _settings()
    frames = _frames([(105,106,104,105)]*2,settings,["A_USDT","B_USDT","C_USDT"])
    result = portfolio_replay(frames,settings,signal_provider=_first_each(stop=99.99,target=100.02))
    assert result["accepted_signals"] == 1 and result["rejected_margin"] == 2
    assert result["peak_reserved_margin_usdt"] == pytest.approx(900)
    assert result["peak_active_orders_positions"] == 1


def test_shared_daily_loss_blocks_all_pairs_and_resets_on_new_utc_date():
    settings = _settings()
    index = pd.date_range("2026-10-05T23:57Z",periods=4,freq="min")
    frames = _frames([(105,106,104,105),(50,51,45,50),(105,106,104,105),(105,106,104,105)],settings,["A_USDT","B_USDT"],index)
    def provider(**kwargs):
        return _setup(kwargs,identifier=f"idea-{kwargs['now']}")
    result = portfolio_replay(frames,settings,signal_provider=provider)
    assert result["final_realized_equity"] == pytest.approx(950)
    assert result["trades_count"] == 2
    assert result["daily_blocked_signal_checks"] == 2
    assert result["accepted_signals"] == 4  # New date admits both symbols again.
    assert result["daily_close"][0]["opening_equity"] == 1000
    assert result["daily_close"][1]["opening_equity"] == 950


def test_daily_limit_uses_opening_equity_even_after_profit_in_same_day():
    settings = _settings()
    gap_price = 100*(1-30.05/50.5)
    frames = _frames([(105,106,104,105),(100,121,95,120),(gap_price,gap_price+1,gap_price-1,gap_price)],settings)
    def provider(**kwargs):
        return _setup(kwargs,identifier=f"idea-{kwargs['now']}")
    result = portfolio_replay(frames,settings,signal_provider=provider)
    assert result["final_realized_equity"] == pytest.approx(979.95)
    assert result["daily_blocked_signal_checks"] == 1
    assert result["accepted_signals"] == 2


def test_all_pairs_exits_settle_before_other_symbol_sizing_at_same_close():
    settings = _settings()
    symbols = ["A_USDT","B_USDT"]
    frames = _frames([(105,106,104,105),(100,121,95,120)],settings,symbols)
    def provider(**kwargs):
        minute = kwargs["now"].minute
        return _setup(kwargs) if (kwargs["symbol"]=="B_USDT" and minute==1 or kwargs["symbol"]=="A_USDT" and minute==2) else None
    result = portfolio_replay(frames,settings,symbols=symbols,signal_provider=provider)
    assert result["final_realized_equity"] == 1010
    assert result["active_orders_positions"][0]["symbol"] == "A_USDT"
    assert result["active_orders_positions"][0]["notional"] == pytest.approx(50.5)


def test_signal_only_at_m5_close_sees_no_future_context_and_cannot_fill_before_confirmation():
    settings = _settings(volium_mode="intraday")
    rows = [(100,121,89,100)]*5 + [(100,105,95,100)]*6
    frames = _frames(rows,settings)
    five = frames["BTC_USDT"]["5m"]
    five.index = pd.date_range("2026-10-05 06:00",periods=len(five),freq="5min",tz="UTC")
    calls = []
    def provider(**kwargs):
        calls.append(kwargs["now"])
        for tf,frame in kwargs["frames"].items():
            assert (frame.index+pd.Timedelta(seconds=SECONDS[tf]) <= kwargs["now"]).all()
        return _setup(kwargs)
    result = portfolio_replay(frames,settings,signal_provider=provider)
    assert calls == [pd.Timestamp("2026-10-05T07:05Z")]
    order = result["active_orders_positions"][0]
    assert order["signal_time"] == order["entry_time"] == "2026-10-05T07:05:00+00:00"
    assert result["trades_count"] == 0 and result["unfinished_positions"] == 1


def test_stable_sweep_id_is_deduplicated_after_trade_closes():
    settings = _settings()
    frames = _frames([(105,106,104,105),(100,121,95,120),(105,106,104,105)],settings)
    result = portfolio_replay(frames,settings,signal_provider=lambda **kwargs:_setup(kwargs))
    assert result["signals"] == result["accepted_signals"] == result["trades_count"] == 1
    assert result["unfinished_limits"] == 0


def test_rejected_idea_can_be_admitted_after_another_pair_exits():
    settings = _settings(max_open_setups=1)
    frames = _frames([(105,106,104,105),(100,110,95,105),(110,121,105,120),(100,110,95,105)],
                     settings,["A_USDT","B_USDT"])
    result = portfolio_replay(frames,settings,signal_provider=lambda **kwargs:_setup(kwargs))
    assert result["signals"] == result["accepted_signals"] == 2
    assert result["rejected_slots"] == 2  # Attempts of the same unconsumed B idea.
    order = result["active_orders_positions"][0]
    assert order["symbol"] == "B_USDT"
    assert order["signal_time"] == order["entry_time"] == "2026-10-05T07:03:00+00:00"


@pytest.mark.parametrize("common_gap",[False,True])
def test_missing_m1_bar_aborts_even_when_all_symbols_share_the_same_hole(common_gap):
    settings = _settings()
    frames = _frames([(105,106,104,105)]*3,settings,["A_USDT","B_USDT"])
    for symbol in (["A_USDT","B_USDT"] if common_gap else ["B_USDT"]):
        frames[symbol]["1m"] = frames[symbol]["1m"].drop(frames[symbol]["1m"].index[1])
    with pytest.raises(ValueError,match="Missing 1 M1 execution bars"):
        portfolio_replay(frames,settings)


def test_pending_ttl_expires_without_consuming_profit_on_unfilled_price_excursion():
    settings = _settings(pending_order_max_age_minutes=1)
    frames = _frames([(105,106,104,105),(105,125,101,110),(100,121,95,120)],settings)
    result = portfolio_replay(frames,settings,signal_provider=_first_each())
    assert result["expired_limits"] == result["unfilled_signals"] == 1
    assert result["trades_count"] == 0 and result["final_realized_equity"] == 1000


@pytest.mark.parametrize("rows,rate,direction",[
    ([(105,106,104,105),(105,125,99,110),(110,121,105,120)],-.001,"LONG"),
    ([(105,106,104,105),(100,125,85,100)],.001,"LONG"),
    ([(105,106,104,105),(100,110,95,105),(105,121,104,120)],.001,"LONG"),
    ([(95,96,94,95),(100,105,90,95),(95,96,79,80)],.001,"SHORT"),
])
def test_single_asset_matches_existing_m1_replay_golden_including_costs_and_funding(rows,rate,direction):
    settings = _settings(paper_fee_bps=5,paper_slippage_bps=2)
    frames = _frames(rows,settings)
    rates = pd.Series([rate],index=pd.DatetimeIndex(["2026-10-05T07:02Z"]))
    options = dict(direction=direction,stop=90 if direction=="LONG" else 110,target=120 if direction=="LONG" else 80)
    actual = portfolio_replay(frames,settings,funding_rates={"BTC_USDT":rates},signal_provider=_first_each(**options))
    expected = replay("BTC_USDT",frames["BTC_USDT"],settings,funding_rates=rates,signal_provider=_first_each(**options))
    for key in ("final_realized_equity","net_pnl_usdt","funding_total_usdt","unrealized_pnl_usdt","marked_equity",
                "max_marked_drawdown_percent","max_realized_drawdown_percent","trades_count","expired_limits",
                "funding_events_charged","funding_events_skipped_favorable"):
        assert actual[key] == pytest.approx(expected[key])
    for trade,reference in zip(actual["trades"],expected["trades"]):
        assert {key:trade[key] for key in reference} == reference


def test_future_dated_signal_is_rejected_instead_of_entered_early():
    settings = _settings()
    frames = _frames([(105,106,104,105)],settings)
    def provider(**kwargs):
        setup = _setup(kwargs)
        setup.timestamp = (kwargs["now"]+pd.Timedelta(minutes=1)).to_pydatetime()
        return setup
    with pytest.raises(ValueError,match="confirmed candle close"):
        portfolio_replay(frames,settings,signal_provider=provider)


def _m5_frames(rows, settings, symbols=("BTC_USDT",)):
    frames = _frames(rows, settings, symbols)
    index = pd.date_range("2026-10-05T07:00Z", periods=len(rows), freq="5min")
    for symbol in symbols:
        frames[symbol]["5m"] = pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close"], dtype=float)
        del frames[symbol]["1m"]  # No fabricated minute path is supplied.
    return frames


@pytest.mark.parametrize("rows,ttl,rate", [
    ([(105,106,104,105),(100,125,85,100)], 30, .001),
    ([(105,106,104,105),(105,125,99,110),(110,121,105,120)], 30, -.001),
    ([(105,106,104,105),(105,125,99,110),(100,125,95,120)], 2, -.001),
    ([(105,106,104,105),(100,121,95,120)], 2, .001),
])
def test_explicit_m5_portfolio_matches_coarse_replay_with_ttl_ambiguities_and_funding(rows, ttl, rate):
    settings = _settings(volium_mode="intraday", pending_order_max_age_minutes=ttl,
                         paper_fee_bps=5, paper_slippage_bps=2)
    frames = _m5_frames(rows, settings)
    rates = pd.Series([rate], index=pd.DatetimeIndex(["2026-10-05T07:06Z"]))
    actual = portfolio_replay(frames, settings, execution_timeframe="5m",
                             funding_rates={"BTC_USDT": rates}, signal_provider=_first_each())
    expected = replay("BTC_USDT", frames["BTC_USDT"], settings, funding_rates=rates,
                      signal_provider=_first_each())
    for key in ("final_realized_equity", "funding_total_usdt", "trades_count", "expired_limits",
                "ambiguous_expiry_limits", "max_marked_drawdown_percent"):
        assert actual[key] == pytest.approx(expected[key])
    for trade, reference in zip(actual["trades"], expected["trades"]):
        assert {key: trade[key] for key in reference} == reference
    assert actual["execution_timeframe"] == "5m" and actual["coarse_execution_proxy"]
    assert actual["execution_minutes"] == len(rows) * 5
    assert actual["execution_bars"] == len(rows)


def test_m5_cannot_fill_on_confirmation_bar_and_uses_closed_context_only():
    settings = _settings(volium_mode="intraday")
    frames = _m5_frames([(100,121,89,100),(105,106,104,105)], settings)
    calls = []
    def provider(**kwargs):
        calls.append(kwargs["now"])
        for tf, frame in kwargs["frames"].items():
            assert (frame.index + pd.Timedelta(seconds=SECONDS[tf]) <= kwargs["now"]).all()
        return _setup(kwargs)
    result = portfolio_replay(frames, settings, execution_timeframe="5m", signal_provider=provider)
    assert calls == [pd.Timestamp("2026-10-05T07:05Z")]
    assert result["trades_count"] == 0 and result["unfinished_limits"] == 1


def test_m5_execution_aborts_missing_bar_and_rejects_m1_scalp_confirmation():
    settings = _settings(volium_mode="intraday")
    frames = _m5_frames([(105,106,104,105)]*3, settings, ["A_USDT", "B_USDT"])
    frames["B_USDT"]["5m"] = frames["B_USDT"]["5m"].drop(frames["B_USDT"]["5m"].index[1])
    with pytest.raises(ValueError, match="Missing 1 M5 execution bars"):
        portfolio_replay(frames, settings, execution_timeframe="5m")
    with pytest.raises(ValueError, match="coarser than the strategy confirmation"):
        portfolio_replay({}, _settings(volium_mode="scalp"), symbols=["BTC_USDT"], execution_timeframe="5m")
