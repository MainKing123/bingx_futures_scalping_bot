import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.models import Base, ExecutionRecord
from app.exchange.client import MEXCAPIError
from app.execution import executor as execution_module
from app.execution.executor import AutoExecutor, HALT_STATES
from app.schemas.setup import TradeSetup


def signal(identifier="signal-a", symbol="BTC_USDT"):
    return TradeSetup(id=identifier, timestamp=datetime.now(timezone.utc), symbol=symbol,
        direction="LONG", setup_type="VOLIUM_INTRADAY", htf_bias="BULLISH",
        entry=100, stop_loss=95, take_profits=[110], risk_reward=2)


def client():
    return SimpleNamespace(
        get_positions=AsyncMock(return_value=[]),
        get_open_orders=AsyncMock(return_value=[]),
        get_balance=AsyncMock(return_value={"equity": 1000, "availableBalance": 1000}),
        place_bracket_order=AsyncMock(return_value={"orderId": "123", "vol": "10"}),
        get_order_by_external_id=AsyncMock(return_value={"orderId": "123", "dealVol": 0, "state": 2}),
        get_stop_orders=AsyncMock(return_value=[]), get_history_positions=AsyncMock(return_value=[]),
        cancel_order=AsyncMock(return_value={"errorCode": 0}),
    )


def executor(fake=None):
    return AutoExecutor(fake or client(), Settings(_env_file=None, auto_execution=True))


async def database(monkeypatch, action):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(execution_module, "SessionLocal", sessions)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        await action(sessions)
    finally:
        await engine.dispose()


async def execution_record(sessions, setup_id):
    async with sessions() as session:
        return await session.get(ExecutionRecord, setup_id)


def filled_order(**overrides):
    return {"orderId": "123", "dealVol": 10, "vol": 10, "state": 3,
            "positionId": "456", "dealAvgPrice": 100, **overrides}


def position(**overrides):
    return {"positionId": "456", "holdVol": 10, "positionType": 1, "state": 1, **overrides}


def protection(**overrides):
    return {"positionId": "456", "state": 1, "isFinished": False,
            "stopLossPrice": 95, "takeProfitPrice": 110, "vol": 10, **overrides}


def test_reservation_is_durable_before_order_creation(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()

        async def create(**kwargs):
            record = await execution_record(sessions, setup.id)
            assert record.state == "PREPARED"
            assert kwargs["external_oid"] == record.external_oid
            assert kwargs["entry"] == setup.entry
            assert kwargs["take_profit"] == 110
            return {"orderId": "123", "vol": 10}

        fake.place_bracket_order.side_effect = create
        assert await worker.execute_setup(setup)
        assert (await execution_record(sessions, setup.id)).state == "ACCEPTED"
        fake.place_bracket_order.assert_awaited_once()

    asyncio.run(database(monkeypatch, run))


def test_disabled_execution_never_creates_journal_or_calls_exchange(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = AutoExecutor(fake, Settings(_env_file=None, auto_execution=False))
        assert not await worker.execute_setup(signal())
        assert await execution_record(sessions, "signal-a") is None
        fake.get_positions.assert_not_awaited()
        fake.place_bracket_order.assert_not_awaited()

    asyncio.run(database(monkeypatch, run))


def test_timeout_create_is_never_retried_after_restart_and_halts_new_entries(monkeypatch):
    async def run(sessions):
        fake = client()
        fake.place_bracket_order.side_effect = TimeoutError("ambiguous timeout")
        setup = signal()
        assert await executor(fake).execute_setup(setup)
        record = await execution_record(sessions, setup.id)
        assert record.state == "UNKNOWN"
        restarted = executor(fake)
        assert await restarted.execute_setup(setup)
        assert not await restarted.execute_setup(signal("different", "ETH_USDT"))
        assert await execution_record(sessions, "different") is None
        fake.place_bracket_order.assert_awaited_once()

    asyncio.run(database(monkeypatch, run))


@pytest.mark.parametrize("error,accepted,state", [
    (MEXCAPIError(2005), False, "REJECTED"),
    (MEXCAPIError(2042), True, "UNKNOWN"),
    (MEXCAPIError(500, uncertain=True), True, "UNKNOWN"),
])
def test_exchange_error_classification_and_durable_no_retry(monkeypatch, error, accepted, state):
    async def run(sessions):
        fake = client()
        fake.place_bracket_order.side_effect = error
        setup = signal()
        assert await executor(fake).execute_setup(setup) is accepted
        assert (await execution_record(sessions, setup.id)).state == state
        assert await executor(fake).execute_setup(setup) is accepted
        fake.place_bracket_order.assert_awaited_once()

    asyncio.run(database(monkeypatch, run))


@pytest.mark.parametrize("cause", ["existing_position", "existing_order", "no_margin", "invalid_bracket"])
def test_preflight_rejection_never_sends_an_order(monkeypatch, cause):
    async def run(sessions):
        fake = client()
        setup = signal()
        if cause == "existing_position":
            fake.get_positions.return_value = [position()]
        elif cause == "existing_order":
            fake.get_open_orders.return_value = [{"symbol": "BTC_USDT", "orderId": "999"}]
        elif cause == "no_margin":
            fake.get_balance.return_value = {"equity": 1000, "availableBalance": 0}
        else:
            setup.stop_loss = 105
        assert not await executor(fake).execute_setup(setup)
        assert (await execution_record(sessions, setup.id)).state == "REJECTED"
        fake.place_bracket_order.assert_not_awaited()

    asyncio.run(database(monkeypatch, run))


def test_entry_size_uses_actual_equity_and_available_margin_cap(monkeypatch):
    async def run(sessions):
        fake = client()
        fake.get_balance.return_value = {"equity": 2000, "availableBalance": 4}
        worker = executor(fake)
        setup = signal()
        assert await worker.execute_setup(setup)
        expected = 4 * worker.settings.default_leverage * 0.9
        assert fake.place_bracket_order.await_args.kwargs["notional_usdt"] == pytest.approx(expected)
        assert setup.position_size_usdt == pytest.approx(expected)

    asyncio.run(database(monkeypatch, run))


def test_concurrent_duplicate_execution_creates_once(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        assert await asyncio.gather(worker.execute_setup(signal()), worker.execute_setup(signal())) == [True, True]
        fake.place_bracket_order.assert_awaited_once()

    asyncio.run(database(monkeypatch, run))


def test_live_fill_stays_active_when_exchange_protection_covers_position(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        fake.get_order_by_external_id.return_value = filled_order()
        fake.get_positions.return_value = [position()]
        fake.get_stop_orders.return_value = [protection()]
        assert await worker.reconcile(setup) is None
        record = await execution_record(sessions, setup.id)
        assert record.state == "FILLED"
        assert record.actual_entry == 100
        fake.get_history_positions.assert_not_awaited()

    asyncio.run(database(monkeypatch, run))


@pytest.mark.parametrize("stops", [[], [protection(vol=9)], [protection(state=2)], [protection(stopLossPrice=105)],
    [protection(stopLossVol=9, takeProfitVol=10)], [protection(stopLossVol=10, takeProfitVol=9)]])
def test_missing_invalid_or_insufficient_protection_halts_entries(monkeypatch, stops):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        fake.get_order_by_external_id.return_value = filled_order()
        fake.get_positions.return_value = [position()]
        fake.get_stop_orders.return_value = stops
        assert await worker.reconcile(setup) is None
        assert (await execution_record(sessions, setup.id)).state == "UNPROTECTED"
        assert not await executor(fake).execute_setup(signal("other", "ETH_USDT"))
        fake.place_bracket_order.assert_awaited_once()

    asyncio.run(database(monkeypatch, run))


def test_unprotected_state_survives_next_reconciliation_network_failure(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        fake.get_order_by_external_id.return_value = filled_order()
        fake.get_positions.return_value = [position()]
        await worker.reconcile(setup)
        fake.get_stop_orders.side_effect = TimeoutError("protection request timed out")
        try:
            await worker.reconcile(setup)
        except TimeoutError:
            pass
        assert (await execution_record(sessions, setup.id)).state in HALT_STATES

    asyncio.run(database(monkeypatch, run))


def test_order_without_position_identity_cannot_silently_skip_protection(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        fake.get_order_by_external_id.return_value = filled_order(positionId=None)
        # No unambiguous position can be inferred from the exchange response.
        fake.get_positions.return_value = []
        await worker.reconcile(setup)
        assert (await execution_record(sessions, setup.id)).state in HALT_STATES

    asyncio.run(database(monkeypatch, run))


def test_realized_close_is_recoverable_after_journal_commit_before_tracker_commit(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        fake.get_order_by_external_id.return_value = filled_order()
        fake.get_history_positions.return_value = [{"positionId": "456", "state": 3,
            "realised": -5.5, "closeAvgPrice": 95}]
        result = await worker.reconcile(setup)
        assert result[0] == "CLOSED"
        assert result[2] == -5.5
        assert (await execution_record(sessions, setup.id)).state == "CLOSED"
        restarted = executor(fake)
        recovered = await restarted.reconcile(setup)
        assert recovered[0] == "CLOSED"
        assert recovered[2] == -5.5
        assert recovered[1] == 95
        fake.get_order_by_external_id.assert_awaited_once()
        fake.place_bracket_order.assert_awaited_once()

    asyncio.run(database(monkeypatch, run))


def test_unconfirmed_order_lookup_never_releases_unknown_reservation(monkeypatch):
    async def run(sessions):
        fake = client()
        fake.place_bracket_order.side_effect = TimeoutError()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        fake.get_order_by_external_id.side_effect = MEXCAPIError(2040)
        assert await worker.reconcile(setup) is None
        assert (await execution_record(sessions, setup.id)).state == "UNKNOWN"
        fake.cancel_order.assert_not_awaited()

    asyncio.run(database(monkeypatch, run))


def test_confirmed_unfilled_cancellation_recovers_after_restart(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        fake.get_order_by_external_id.return_value = {"orderId": "123", "dealVol": 0, "state": 4}
        assert await worker.reconcile(setup) == ("CANCELLED", setup.entry, 0.0)
        assert await executor(fake).reconcile(setup) == ("CANCELLED", setup.entry, 0.0)
        fake.get_order_by_external_id.assert_awaited_once()

    asyncio.run(database(monkeypatch, run))


def test_cancel_filled_entry_rejects_without_mutation(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        fake.get_order_by_external_id.return_value = filled_order(dealVol=1)
        with pytest.raises(ValueError, match="Filled"):
            await worker.cancel_pending(setup.id)
        fake.cancel_order.assert_not_awaited()

    asyncio.run(database(monkeypatch, run))


def test_pending_timeout_cancel_requires_exchange_confirmation(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        async with sessions() as session:
            record = await session.get(ExecutionRecord, setup.id)
            record.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
            await session.commit()
        assert await worker.reconcile(setup) is None
        fake.cancel_order.assert_awaited_once_with("BTC_USDT", "123")
        assert (await execution_record(sessions, setup.id)).state not in {"CLOSED", "REJECTED"}

    asyncio.run(database(monkeypatch, run))


def test_partial_fill_cancels_residual_once_and_retains_reservation_until_terminal(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        fake.get_order_by_external_id.return_value = filled_order(dealVol=5, state=2)
        fake.get_positions.return_value = [position(holdVol=5)]
        fake.get_stop_orders.return_value = [protection(stopLossVol=5, takeProfitVol=5)]
        assert await worker.reconcile(setup) is None
        record = await execution_record(sessions, setup.id)
        assert record.cancel_state == "REQUESTED"
        assert record.state in HALT_STATES
        assert record.volume == 5
        assert await executor(fake).reconcile(setup) is None
        fake.cancel_order.assert_awaited_once_with("BTC_USDT", "123")
        # A closed partial position cannot release risk while a residual entry may fill again.
        fake.get_positions.return_value = []
        fake.get_history_positions.return_value = [{"positionId": "456", "state": 3,
            "realised": 4, "closeAvgPrice": 108}]
        assert await worker.reconcile(setup) is None
        fake.get_history_positions.assert_not_awaited()
        # The exchange confirms residual cancellation before closure becomes final.
        fake.get_order_by_external_id.return_value = filled_order(dealVol=5, state=4)
        assert await worker.reconcile(setup) == ("CLOSED", 108, 4)
        record = await execution_record(sessions, setup.id)
        assert record.cancel_state == "CONFIRMED"
        assert record.state == "CLOSED"
        fake.cancel_order.assert_awaited_once()

    asyncio.run(database(monkeypatch, run))


def test_residual_cancel_timeout_is_durable_and_never_retried_on_restart(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        fake.get_order_by_external_id.return_value = filled_order(dealVol=5, state=2)
        fake.cancel_order.side_effect = TimeoutError("cancel outcome unknown")
        with pytest.raises(TimeoutError):
            await worker.reconcile(setup)
        assert (await execution_record(sessions, setup.id)).cancel_state == "REQUESTED"
        fake.get_positions.return_value = [position(holdVol=5)]
        fake.get_stop_orders.return_value = [protection(stopLossVol=5, takeProfitVol=5)]
        assert await executor(fake).reconcile(setup) is None
        fake.cancel_order.assert_awaited_once()
        assert (await execution_record(sessions, setup.id)).state in HALT_STATES

    asyncio.run(database(monkeypatch, run))


def test_expired_unfilled_cancel_mutation_is_not_repeated_after_restart(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        async with sessions() as session:
            record = await session.get(ExecutionRecord, setup.id)
            record.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
            await session.commit()
        await worker.reconcile(setup)
        await executor(fake).reconcile(setup)
        fake.cancel_order.assert_awaited_once()
        assert (await execution_record(sessions, setup.id)).cancel_state == "REQUESTED"

    asyncio.run(database(monkeypatch, run))


def test_manual_cancel_timeout_is_persisted_and_not_retried_after_restart(monkeypatch):
    async def run(sessions):
        fake = client()
        worker = executor(fake)
        setup = signal()
        await worker.execute_setup(setup)
        fake.cancel_order.side_effect = TimeoutError("cancel outcome unknown")
        with pytest.raises(TimeoutError):
            await worker.cancel_pending(setup.id)
        record = await execution_record(sessions, setup.id)
        assert record.cancel_state == "REQUESTED"
        assert record.state in HALT_STATES
        fake.cancel_order.side_effect = None
        await executor(fake).cancel_pending(setup.id)
        fake.cancel_order.assert_awaited_once()

    asyncio.run(database(monkeypatch, run))


def test_accepted_order_response_parse_failure_preserves_unknown_reservation(monkeypatch):
    async def run(sessions):
        fake = client()
        fake.place_bracket_order.return_value = {"orderId": "123", "vol": 10, "price": "malformed"}
        setup = signal()
        assert await executor(fake).execute_setup(setup)
        record = await execution_record(sessions, setup.id)
        assert record.state == "UNKNOWN"
        assert record.order_id == "123"
        assert await executor(fake).execute_setup(setup)
        fake.place_bracket_order.assert_awaited_once()

    asyncio.run(database(monkeypatch, run))


def test_contract_validation_before_create_can_definitively_reject(monkeypatch):
    async def run(sessions):
        fake = client()
        fake.place_bracket_order.side_effect = ValueError("below contract minimum")
        setup = signal()
        assert not await executor(fake).execute_setup(setup)
        assert (await execution_record(sessions, setup.id)).state == "REJECTED"
        assert not await executor(fake).execute_setup(setup)
        fake.place_bracket_order.assert_awaited_once()

    asyncio.run(database(monkeypatch, run))
