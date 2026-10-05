"""Cash-risk, restart and leverage regressions; no account/network calls."""
from datetime import datetime, timezone
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.runtime_settings import RuntimeSettings
from app.db.repository import record_to_setup
from app.execution.economics import prepare_runtime_setup, trade_economics, enforce_economics, current_contract_with_basis
from app.schemas.setup import TradeSetup


def setup(symbol="BTC_USDT", direction="LONG", stop_distance=.8):
    sign = 1 if direction == "LONG" else -1
    return TradeSetup(id="test", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        symbol=symbol, direction=direction, setup_type="VOLIUM_INTRADAY",
        htf_bias="BULLISH" if sign == 1 else "BEARISH", entry=100,
        stop_loss=100-sign*stop_distance, take_profits=[100+sign*2*stop_distance], risk_reward=2)


def contract(**updates):
    return {"apiAllowed":True, "state":0, "positionOpenType":3,
            "contractSize":.01, "volUnit":1, "minVol":1, "maxVol":1000000,
            "limitMaxVol":1000000, "priceUnit":.0001, "priceScale":4, "volScale":0,
            "minLeverage":1, "maxLeverage":100, "countryConfigContractMaxLeverage":0,
            "riskLimitMode":"CUSTOM", "riskLimitType":"BY_VOLUME",
            "riskLimitCustom":[{"level":1,"maxVol":1000000,"mmr":.001,"imr":.01,"maxLeverage":100}],
            "liquidationFeeRate":.0004, "priceCoefficientVariation":0, "takerFeeRate":.0002,
            "feeRateMode":"NORMAL", **updates}


def policy(**updates):
    return RuntimeSettings(_env_file=None, **updates)


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_fee_inclusive_cash_risk_does_not_grow_with_leverage(direction):
    idea=setup(direction=direction)
    high=prepare_runtime_setup(idea,policy(),contract(),1000,900)
    low=prepare_runtime_setup(idea,policy(default_leverage=10),contract(),1000,900)
    assert high.leverage==50 and low.leverage==10
    assert high.position_size_usdt==low.position_size_usdt
    assert high.economics["modeled_stop_loss_usdt"] <= 5
    assert high.economics["initial_margin_usdt"] * 5 == pytest.approx(low.economics["initial_margin_usdt"])
    assert high.economics["modeled_target_profit_usdt"] == low.economics["modeled_target_profit_usdt"]
    assert idea.stop_loss == (99.2 if direction=="LONG" else 100.8)


def test_tight_stop_is_rejected_instead_of_ignoring_costs():
    tight=setup(stop_distance=.1)
    economics=trade_economics(tight,5,2)
    assert economics.net_reward_risk < .3
    with pytest.raises(ValueError,match="Net reward/risk"):
        enforce_economics(tight,policy())
    assert tight.stop_loss==99.9


def test_wide_stop_is_rejected_instead_of_moved():
    idea=setup(stop_distance=2)
    with pytest.raises(ValueError,match="Structural stop"):
        prepare_runtime_setup(idea,policy(),contract(),1000,900)
    assert idea.stop_loss==98


def test_contract_tier_country_and_asset_caps_are_enforced():
    limited=contract(countryConfigContractMaxLeverage=30,
        riskLimitCustom=[{"maxVol":1000000,"mmr":.001,"imr":.05,"maxLeverage":20}])
    prepared=prepare_runtime_setup(setup(),policy(),limited,1000,900)
    assert prepared.leverage==20
    sol=prepare_runtime_setup(setup("SOL_USDT"),policy(),contract(),1000,900)
    other=prepare_runtime_setup(setup("DOGE_USDT"),policy(),contract(),1000,900)
    assert sol.leverage==25 and other.leverage==10


def test_leverage_reduces_when_liquidation_proxy_is_too_close():
    prepared=prepare_runtime_setup(setup(stop_distance=1.4),policy(),
        contract(observedFairLastBasisFraction=.004),1000,900)
    assert prepared.leverage==25
    assert prepared.economics["stop_and_gap_allowance_fraction"] < .8*prepared.economics["conservative_liquidation_distance_fraction"]


def test_no_admissible_leverage_rejects_order():
    with pytest.raises(ValueError,match="No leverage"):
        prepare_runtime_setup(setup(),policy(),contract(
            riskLimitCustom=[{"maxVol":1000000,"mmr":.095,"imr":.1,"maxLeverage":10}]),1000,900)


def test_missing_or_exceeded_risk_tier_is_rejected():
    with pytest.raises(ValueError,match="Missing custom"):
        prepare_runtime_setup(setup(),policy(),contract(riskLimitCustom=[]),1000,900)
    with pytest.raises(ValueError,match="exceeds published"):
        prepare_runtime_setup(setup(),policy(),contract(
            riskLimitCustom=[{"maxVol":10,"mmr":.001,"imr":.01,"maxLeverage":100}]),1000,900)


def test_incremental_tiers_use_position_size_and_initial_margin_rate():
    prepared=prepare_runtime_setup(setup(),policy(),contract(
        riskLimitMode="INCREASE", riskBaseVol=100, riskIncrVol=100,
        riskLevelLimit=10, maintenanceMarginRate=.001, initialMarginRate=.01,
        riskIncrMmr=.001, riskIncrImr=.005),1000,900)
    assert prepared.leverage<=28
    assert prepared.economics["maintenance_margin_rate"]>.001


def test_lot_rounding_margin_and_higher_public_fee_stay_conservative():
    prepared=prepare_runtime_setup(setup(),policy(),contract(takerFeeRate=.0007),1000,10)
    assert prepared.economics["roundtrip_cost_fraction"]==pytest.approx(.0009*(1+prepared.take_profits[0]/prepared.entry))
    assert prepared.economics["modeled_stop_loss_usdt"]<=5
    assert prepared.economics["reserved_margin_usdt"]<=10
    assert prepared.position_size_usdt % 1==pytest.approx(0)


def test_restart_preserves_leverage_and_economics():
    prepared=prepare_runtime_setup(setup(),policy(),contract(),1000,900)
    restored=record_to_setup(SimpleNamespace(payload=prepared.model_dump_json(),status="ACTIVE"))
    assert restored.leverage==prepared.leverage
    assert restored.economics==prepared.economics
    old=record_to_setup(SimpleNamespace(payload=setup().model_dump_json(),status="ACTIVE"))
    assert not hasattr(old,"leverage")


@pytest.mark.parametrize("bad", [float("nan"),float("inf"),-1,True])
def test_invalid_cost_is_never_admitted(bad):
    with pytest.raises(ValueError):
        trade_economics(setup(),bad,2)


def test_frozen_settings_stays_3x_and_current_runtime_accepts_50x():
    assert Settings(_env_file=None).default_leverage==3
    assert policy().default_leverage==50
    with pytest.raises(ValueError):
        policy(default_leverage=51)


def test_contract_variation_coefficient_is_not_treated_as_observed_40_percent_basis():
    prepared=prepare_runtime_setup(setup("DOGE_USDT"),policy(),contract(priceCoefficientVariation=.4),1000,900)
    assert prepared.leverage==10
    assert prepared.economics["fair_last_basis_buffer_fraction"]==.002
    assert prepared.economics["fair_last_basis_observed"] is False


def test_unvalidated_v5_profile_never_enables_live_or_unsupported_swing():
    with pytest.raises(ValueError,match="paper trading only"):
        policy(auto_execution=True)
    with pytest.raises(ValueError,match="intraday/scalp"):
        policy(volium_mode="swing")


@pytest.mark.parametrize("mutation,reason", [({"timestamp":1},"stale"),({"fairPrice":float("nan")},"fair price"),
                                             ({"symbol":"ETH_USDT"},"another contract")])
def test_stale_invalid_or_wrong_symbol_mark_observation_is_not_used(mutation,reason):
    ticker={"symbol":"BTC_USDT","lastPrice":100,"fairPrice":100.01,
            "timestamp":int(datetime.now(timezone.utc).timestamp()*1000),**mutation}
    client=SimpleNamespace(get_contract=AsyncMock(return_value=contract()),get_ticker=AsyncMock(return_value=ticker))
    with pytest.raises(ValueError,match=reason):
        asyncio.run(current_contract_with_basis(client,"BTC_USDT"))
