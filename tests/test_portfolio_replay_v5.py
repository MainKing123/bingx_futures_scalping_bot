"""Independent financial/casual fixtures, without exchange or outcome searches."""
import pandas as pd
import pytest

from app.portfolio_replay_v5 import portfolio_replay_v5, binance_proxy_admission, reservation
from app.runtime_settings import RuntimeSettings
from app.schemas.setup import TradeSetup
from app.execution.economics import RuntimeTradeSetup, trade_economics
from app.replay import required_timeframes, SECONDS


def settings(**kwargs):
    return RuntimeSettings(_env_file=None, mexc_api_key="",mexc_api_secret="",auto_execution=False,
        volium_mode="scalp",volium_session_enabled=False,account_balance_usdt=1000,
        risk_per_trade_percent=kwargs.pop("risk_per_trade_percent",.5),paper_fee_bps=kwargs.pop("paper_fee_bps",5),
        paper_slippage_bps=kwargs.pop("paper_slippage_bps",2),**kwargs)


def frames(rows, config, symbols=("BTC_USDT",), *, freq="min"):
    index=pd.date_range("2026-10-05T07:00Z",periods=len(rows),freq=freq)
    entry=pd.DataFrame(rows,index=index,columns=["open","high","low","close"],dtype=float)
    output={}
    for symbol in symbols:
        output[symbol]={"1m" if freq=="min" else "5m":entry.copy()}
        for tf in required_timeframes(config):
            if tf in output[symbol]:continue
            times=pd.date_range(end=index[0]+pd.Timedelta(days=1),periods=150,freq=f"{SECONDS[tf]}s")
            output[symbol][tf]=pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.},index=times)
    return output


def first_signal(**kwargs):
    seen=set()
    def provider(**args):
        if args["symbol"] in seen:return None
        seen.add(args["symbol"])
        sign=1 if kwargs.get("direction","LONG")=="LONG" else -1
        distance=kwargs.get("distance",.8)
        return TradeSetup(id="idea",symbol=args["symbol"],timestamp=args["now"],
            setup_type=f"VOLIUM_{args['mode'].upper()}",direction="LONG" if sign==1 else "SHORT",
            htf_bias="BULLISH" if sign==1 else "BEARISH",entry=100,
            stop_loss=100-sign*distance,take_profits=[100+sign*2*distance],risk_reward=2)
    return provider


def admission(leverage=20):
    def prepare(setup, config, *, equity, available_margin):
        economics=trade_economics(setup,config.paper_fee_bps,config.paper_slippage_bps)
        per_entry=(config.paper_fee_bps+config.paper_slippage_bps)/10000
        n=min(equity*config.risk_per_trade_percent/100/economics.modeled_stop_loss_fraction,
              available_margin/(1/leverage+per_entry))
        return RuntimeTradeSetup.model_validate({**setup.model_dump(),"leverage":leverage,
            "position_size_usdt":n,"economics":{"modeled_fee_bps_per_side":config.paper_fee_bps,
            "modeled_slippage_bps_per_side":config.paper_slippage_bps}})
    return prepare


def test_costs_are_paid_at_fill_and_actual_exit_notional_without_double_count():
    config=settings()
    data=frames([(102,102.1,101.9,102),(100,100.2,99.7,100.1),(101,101.7,100.9,101.6)],config)
    result=portfolio_replay_v5(data,config,signal_provider=first_signal(),admission=admission())
    trade=result["trades"][0]; n=trade["notional"]
    assert trade["status"]=="TP1_HIT"
    assert trade["entry_fee_usdt"]==pytest.approx(n*.0005)
    assert trade["exit_fee_usdt"]==pytest.approx(n*1.016*.0005)
    assert trade["entry_slippage_usdt"]==pytest.approx(n*.0002)
    assert trade["gross_price_pnl_usdt"]==pytest.approx(n*.016)
    assert result["net_pnl_usdt"]==pytest.approx(n*(.016-.0007*(1+1.016)))
    assert trade["pnl_usdt"]==pytest.approx(result["net_pnl_usdt"])
    assert result["fees_total_usdt"]+result["slippage_total_usdt"]==pytest.approx(n*.0007*(1+1.016))
    assert trade["margin_roi_percent"]==pytest.approx(trade["pnl_usdt"]/(n/20)*100)


def test_entry_cost_is_realized_before_position_exit_and_pending_has_no_fee():
    config=settings()
    pending=portfolio_replay_v5(frames([(102,103,101,102)]*2,config),config,
        signal_provider=first_signal(),admission=admission())
    assert pending["final_realized_equity"]==1000
    order=pending["active_orders_positions"][0]
    assert order["margin_reserved_usdt"]==pytest.approx(order["notional"]*(1/20+.0007))
    filled=portfolio_replay_v5(frames([(102,103,101,102),(100,100.1,99.9,100)],config),config,
        signal_provider=first_signal(),admission=admission())
    order=filled["active_orders_positions"][0]; n=order["notional"]
    assert filled["unfinished_positions"]==1
    assert filled["final_realized_equity"]==pytest.approx(1000-n*.0007)
    assert order["margin_reserved_usdt"]==pytest.approx(n/20)
    assert filled["open_entry_cost_usdt"]==pytest.approx(n*.0007)
    assert sum(value["realized_pnl_usdt"] for value in filled["symbol_results"])==pytest.approx(filled["net_pnl_usdt"])


@pytest.mark.parametrize("direction",["LONG","SHORT"])
def test_fee_inclusive_stop_loss_does_not_exceed_point_five_percent(direction):
    config=settings()
    end=(99.2,99.3,99.1,99.2) if direction=="LONG" else (100.8,100.9,100.7,100.8)
    data=frames([(100,100.1,99.9,100),(100,100.2,99.7,100),end],config)
    result=portfolio_replay_v5(data,config,signal_provider=first_signal(direction=direction),admission=admission())
    assert result["trades_count"]==1
    assert -result["trades"][0]["pnl_usdt"]<=5+1e-8
    assert result["gap_losses_beyond_planned_risk"]==0


def test_gap_loss_can_exceed_cash_risk_is_reported_and_not_clipped():
    config=settings()
    data=frames([(102,103,101,102),(100,100.1,99.9,100),(90,91,89,90)],config)
    result=portfolio_replay_v5(data,config,signal_provider=first_signal(),admission=admission())
    assert result["trades"][0]["exit"]==90
    assert result["gap_losses_beyond_planned_risk"]==1
    assert result["net_pnl_usdt"]<-5
    assert result["exchange_liquidations_reconstructed"] is False


def test_funding_cash_is_applied_once_and_reconciles_with_trade_cash():
    config=settings()
    data=frames([(102,103,101,102),(100,100.1,99.9,100),(100,100.1,99.9,100),(101,101.7,100.9,101.6)],config)
    rate=pd.Series([.001],index=pd.DatetimeIndex(["2026-10-05T07:02:30Z"]))
    result=portfolio_replay_v5(data,config,signal_provider=first_signal(),admission=admission(),funding_rates={"BTC_USDT":rate})
    trade=result["trades"][0]
    assert result["funding_events_charged"]==1
    assert result["funding_total_usdt"]==pytest.approx(-trade["notional"]*.001)
    assert result["net_pnl_usdt"]==pytest.approx(trade["pnl_usdt"])
    assert trade["price_pnl_usdt"]+trade["funding_usdt"]==pytest.approx(trade["pnl_usdt"])


def test_leverage_changes_margin_without_changing_cash_risk_and_returns():
    config=settings()
    data=frames([(102,103,101,102),(100,100.2,99.7,100),(101,101.7,100.9,101.6)],config)
    low=portfolio_replay_v5(data,config,signal_provider=first_signal(),admission=admission(10))
    high=portfolio_replay_v5(data,config,signal_provider=first_signal(),admission=admission(50))
    assert low["net_pnl_usdt"]==high["net_pnl_usdt"]
    assert low["trades"][0]["notional"]==high["trades"][0]["notional"]
    assert low["peak_reserved_margin_usdt"]>high["peak_reserved_margin_usdt"]*4.8


def test_bad_admission_cannot_borrow_margin_or_increase_cash_risk():
    config=settings()
    data=frames([(102,103,101,102)]*2,config)
    def corrupt(setup,config,**kwargs):
        return admission()(setup,config,**kwargs).model_copy(update={"position_size_usdt":1000000})
    with pytest.raises(ValueError,match="unavailable shared margin"):
        portfolio_replay_v5(data,config,signal_provider=first_signal(),admission=corrupt)


def test_admission_rejection_preserves_signal_count_and_cash():
    config=settings()
    result=portfolio_replay_v5(frames([(102,103,101,102)]*2,config),config,
        signal_provider=first_signal(distance=.1),admission=binance_proxy_admission({"BTC_USDT":{"liquidationFee":.0125}}))
    assert result["signals"]==1 and result["accepted_signals"]==0
    assert result["rejected_admission"]==1 and result["net_pnl_usdt"]==0
    assert any("Net reward/risk" in reason for reason in result["admission_rejections_by_reason"])


def test_binance_proxy_does_not_apply_mexc_placeholder_tiers_and_uses_fee_guard():
    config=settings()
    args={"now":pd.Timestamp("2026-01-01",tz="UTC"),"symbol":"BTC_USDT","mode":"scalp"}
    idea=first_signal()(**args)
    prepare=binance_proxy_admission({"BTC_USDT":{"liquidationFee":".0125","maintMarginPercent":"2.5"}})
    result=prepare(idea,config,equity=1000,available_margin=900)
    assert result.leverage==25
    assert result.economics["maintenance_margin_rate"]==.01
    assert result.economics["historical_tick_lot_tiers_not_applied"] is True
    assert result.economics["modeled_stop_loss_usdt"]<=5


def test_no_target_credit_on_ambiguous_fill_and_expiry_cancels_without_cost():
    config=settings(pending_order_max_age_minutes=1)
    data=frames([(102,103,101,102),(102,103,101,102),(100,102,99.9,101)],config)
    result=portfolio_replay_v5(data,config,signal_provider=first_signal(),admission=admission())
    assert result["expired_limits"]==1 and result["trades_count"]==0
    assert result["fees_total_usdt"]==0


def test_coarse_intraday_execution_is_explicit_and_causal():
    config=settings(); config.volium_mode="intraday"
    data=frames([(102,103,101,102),(100,100.2,99.7,100),(101,101.7,100.9,101.6)],config,freq="5min")
    seen=[]
    one=first_signal()
    def provider(**kwargs):
        seen.append(kwargs)
        for tf,frame in kwargs["frames"].items():
            assert (frame.index+pd.Timedelta(seconds=SECONDS[tf])<=kwargs["now"]).all()
        return one(**kwargs)
    result=portfolio_replay_v5(data,config,signal_provider=provider,admission=admission(),execution_timeframe="5m")
    assert result["coarse_execution_proxy"] is True
    assert result["execution_bars"]==3 and result["execution_minutes"]==15
    assert result["trades"][0]["entry_time"]>=seen[0]["now"].isoformat()


def test_missing_execution_bar_aborts_before_any_provider_or_cash():
    config=settings(); data=frames([(102,103,101,102)]*4,config)
    data["BTC_USDT"]["1m"]=data["BTC_USDT"]["1m"].drop(data["BTC_USDT"]["1m"].index[1])
    with pytest.raises(ValueError,match="Missing 1 M1"):
        portfolio_replay_v5(data,config,admission=admission())


def test_shared_wallet_uses_each_position_leverage_and_pending_margin_capacity():
    config=settings(paper_fee_bps=0,paper_slippage_bps=0,risk_per_trade_percent=5)
    data=frames([(102,103,101,102)]*2,config,symbols=("BTC_USDT","ETH_USDT"))
    def mixed(setup,config,**kwargs):
        return admission(10 if setup.symbol=="BTC_USDT" else 50)(setup,config,**kwargs)
    result=portfolio_replay_v5(data,config,signal_provider=first_signal(),admission=mixed)
    assert [order["leverage"] for order in result["active_orders_positions"]]==[10,50]
    assert result["peak_reserved_margin_usdt"]==pytest.approx(625+125)
    capped=portfolio_replay_v5(data,config,signal_provider=first_signal(),admission=admission(10))
    assert [order["notional"] for order in capped["active_orders_positions"]]==pytest.approx([6250,2750])
    assert capped["peak_reserved_margin_usdt"]==pytest.approx(900)
    assert capped["downsized_for_margin"]==1


def test_unrealized_inventory_loss_reduces_available_margin_before_later_signal():
    config=settings(paper_fee_bps=0,paper_slippage_bps=0)
    data=frames([(102,103,101,102),(100,100.1,99.25,99.3),(99.3,99.4,99.25,99.3)],config,
                symbols=("BTC_USDT","ETH_USDT"))
    first=first_signal()
    def provider(**kwargs):
        if kwargs["symbol"]=="ETH_USDT" and kwargs["now"]!=pd.Timestamp("2026-10-05T07:03Z"):
            return None
        return first(**kwargs)
    captured=[]
    def admit(setup,config,**kwargs):
        if setup.symbol=="ETH_USDT":captured.append(kwargs)
        return admission(10)(setup,config,**kwargs)
    portfolio_replay_v5(data,config,signal_provider=provider,admission=admit)
    assert captured[0]["equity"]==1000
    assert captured[0]["available_margin"]==pytest.approx(.9*(1000-625*.007)-62.5)


def test_entry_cash_cost_can_trigger_shared_daily_block_before_exit():
    config=settings(daily_loss_limit_percent=.02)
    data=frames([(102,103,101,102),(100,100.1,99.9,100),(100,100.1,99.9,100)],config,
                symbols=("BTC_USDT","ETH_USDT"))
    first=first_signal()
    def provider(**kwargs):
        if kwargs["symbol"]=="ETH_USDT" and kwargs["now"]<pd.Timestamp("2026-10-05T07:02Z"):
            return None
        return first(**kwargs)
    result=portfolio_replay_v5(data,config,signal_provider=provider,admission=admission())
    assert result["unfinished_positions"]==1 and result["accepted_signals"]==1
    assert result["daily_blocked_signal_checks"]==2
