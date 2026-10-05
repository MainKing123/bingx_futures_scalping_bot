import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pandas as pd

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import Base, SetupRecord, ExecutionRecord
from app.db.repository import record_to_setup
from app.risk.risk_manager import RiskManager
from app.scanner import scanner as scanner_module
from app.scanner.scanner import SignalScanner
from app.execution import executor as executor_module
from app.execution.executor import AutoExecutor
from app.execution.economics import prepare_runtime_setup
from test_execution_economics import setup, contract, policy


def quote(symbol):
    return {"symbol":symbol,"lastPrice":100,"fairPrice":100.01,
            "timestamp":int(datetime.now(timezone.utc).timestamp()*1000)}


def isolated_database(monkeypatch, action):
    async def run():
        engine=create_async_engine("sqlite+aiosqlite:///:memory:")
        sessions=async_sessionmaker(engine,expire_on_commit=False)
        monkeypatch.setattr(scanner_module,"SessionLocal",sessions)
        monkeypatch.setattr(executor_module,"SessionLocal",sessions)
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        try:
            await action(sessions)
        finally:
            await engine.dispose()
    asyncio.run(run())


def test_paper_admission_persists_each_asset_leverage_and_costs(monkeypatch):
    async def action(sessions):
        settings=policy(pair_selection="fixed")
        client=SimpleNamespace(get_contract=AsyncMock(return_value=contract()),
            get_ticker=AsyncMock(side_effect=lambda symbol:quote(symbol)))
        tracker=SimpleNamespace(add=AsyncMock())
        scanner=SignalScanner(client,settings,RiskManager(settings),tracker,None)
        for symbol,expected in (("BTC_USDT",50),("SOL_USDT",25),("DOGE_USDT",10)):
            idea=setup(symbol).model_copy(update={"id":symbol})
            assert await scanner._publish_setup(idea)
            async with sessions() as session:
                saved=record_to_setup(await session.get(SetupRecord,symbol))
                assert saved.leverage==expected
                assert saved.economics["modeled_stop_loss_usdt"]<=5
        async with sessions() as session:
            rows=(await session.scalars(select(SetupRecord))).all()
            reserved=sum(record_to_setup(row).economics["reserved_margin_usdt"] for row in rows)
            assert reserved<=900
        assert tracker.add.await_count==3
    isolated_database(monkeypatch,action)


def test_economically_bad_signal_does_not_reserve_slot_or_publish(monkeypatch):
    async def action(sessions):
        settings=policy(pair_selection="fixed")
        tracker=SimpleNamespace(add=AsyncMock())
        risk=RiskManager(settings)
        scanner=SignalScanner(SimpleNamespace(get_contract=AsyncMock(return_value=contract()),
            get_ticker=AsyncMock(side_effect=lambda symbol:quote(symbol))),settings,risk,tracker,None)
        assert not await scanner._publish_setup(setup(stop_distance=.1))
        assert risk.open_setups==0
        assert "Net reward/risk" in scanner.last_rejections["BTC_USDT"]
        tracker.add.assert_not_awaited()
        async with sessions() as session:
            assert (await session.scalars(select(SetupRecord))).all()==[]
    isolated_database(monkeypatch,action)


def test_paper_cannot_spend_unrealized_loss_from_another_position(monkeypatch):
    async def action(sessions):
        settings=policy(pair_selection="fixed")
        tracker=SimpleNamespace(add=AsyncMock(),marked_equity=10)
        risk=RiskManager(settings)
        scanner=SignalScanner(SimpleNamespace(get_contract=AsyncMock(return_value=contract()),
            get_ticker=AsyncMock(side_effect=lambda symbol:quote(symbol))),settings,risk,tracker,None)
        assert await scanner._publish_setup(setup())
        async with sessions() as session:
            saved=record_to_setup(await session.get(SetupRecord,"test"))
            assert saved.economics["reserved_margin_usdt"]<=9
            assert saved.economics["modeled_stop_loss_usdt"]<=5
    isolated_database(monkeypatch,action)


@pytest.mark.parametrize("mode,entry_tf,expected", [("intraday","5m",991),("scalp","1m",431)])
def test_v5_scanner_fetches_same_event_prefix_as_registered_replay(monkeypatch,mode,entry_tf,expected):
    from app.strategy import volium_v5
    async def action():
        settings=policy(pair_selection="fixed",volium_mode=mode)
        now=datetime(2026,1,2,12,tzinfo=timezone.utc)
        frame=pd.DataFrame({"open":[100],"high":[101],"low":[99],"close":[100]},
            index=pd.DatetimeIndex([now]))
        client=SimpleNamespace(get_klines=AsyncMock(return_value=frame))
        scanner=SignalScanner(client,settings,RiskManager(settings),SimpleNamespace(),None)
        monkeypatch.setattr(volium_v5,"analyze_volium_v5_from_df",lambda **kwargs:None)
        await scanner._scan_symbol("BTC_USDT",mode,now)
        requests={call.args[1]:call.kwargs["limit"] for call in client.get_klines.await_args_list}
        assert requests[entry_tf]==expected
        assert all(limit==110 for tf,limit in requests.items() if tf!=entry_tf)
    asyncio.run(action())


def test_live_preflight_passes_chosen_leverage_and_cash_risk_to_adapter(monkeypatch):
    async def action(sessions):
        settings=policy(auto_execution=True,pair_selection="fixed",volium_strategy_profile="v1_guarded")
        idea=prepare_runtime_setup(setup(),settings,contract(),1000,900)
        async with sessions() as session:
            session.add(SetupRecord(id=idea.id,created_at=idea.timestamp,symbol=idea.symbol,
                status="ACTIVE",execution_mode="live",payload=idea.model_dump_json()))
            await session.commit()
        client=SimpleNamespace(get_positions=AsyncMock(return_value=[]),
            get_open_orders=AsyncMock(return_value=[]),get_balance=AsyncMock(return_value={"equity":1000,"availableBalance":1000}),
            get_contract=AsyncMock(return_value=contract()),get_ticker=AsyncMock(side_effect=lambda symbol:quote(symbol)),
            place_bracket_order=AsyncMock(return_value={"orderId":"mock"}))
        executor=AutoExecutor(client,settings)
        assert await executor.execute_setup(idea)
        call=client.place_bracket_order.await_args.kwargs
        assert call["leverage"]==50
        assert call["notional_usdt"]*idea.economics["modeled_stop_loss_fraction"]<=5
        async with sessions() as session:
            saved=record_to_setup(await session.get(SetupRecord,idea.id))
            assert saved.leverage==50
            assert (await session.get(ExecutionRecord,idea.id)).state=="ACCEPTED"
    isolated_database(monkeypatch,action)
