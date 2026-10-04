import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.models import Base, ExecutionRecord, SetupRecord
from app.db.repository import save_setup, update_setup_status
from app.execution import executor as execution_module
from app.risk.risk_manager import RiskManager
from app.risk import risk_manager as risk_module
from app.schemas.setup import TradeSetup
from app.tracking import trade_tracker as tracker_module
from app.tracking.trade_tracker import TradeTracker


SIGNAL_TIME = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def signal(identifier="signal-a", timestamp=SIGNAL_TIME):
    return TradeSetup(id=identifier, timestamp=timestamp, symbol="BTC_USDT", direction="LONG",
        setup_type="VOLIUM_INTRADAY", htf_bias="BULLISH", entry=100, stop_loss=95,
        take_profits=[110], risk_reward=2, position_size_usdt=100)


def bars(*rows):
    return pd.DataFrame(
        [{"open": o, "high": h, "low": low, "close": c} for _, o, h, low, c in rows],
        index=pd.DatetimeIndex([SIGNAL_TIME + timedelta(minutes=i) for i, *_ in rows]),
    )


def tracker(auto=False):
    settings = Settings(_env_file=None, auto_execution=auto, paper_fee_bps=0, paper_slippage_bps=0)
    client = SimpleNamespace(get_klines=AsyncMock(), get_balance=AsyncMock(return_value={"equity": 1000}))
    executor = SimpleNamespace(reconcile=AsyncMock(return_value=None), cancel_pending=AsyncMock())
    risk = RiskManager(settings)
    ws = SimpleNamespace(broadcast=AsyncMock())
    return TradeTracker(client, settings, risk, executor, ws)


async def database(monkeypatch, action):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(tracker_module, "SessionLocal", sessions)
    monkeypatch.setattr(execution_module, "SessionLocal", sessions)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        await action(sessions)
    finally:
        await engine.dispose()


async def add(worker, sessions, setup=None, mode=None):
    setup = setup or signal()
    async with sessions() as session:
        await save_setup(session, setup, mode or ("live" if worker.settings.auto_execution else "paper"))
    await worker.add(setup)
    worker.risk_manager.open_setups += 1
    return setup


def test_paper_target_touch_before_limit_fill_does_not_close_trade(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)
        await worker.process_paper_bars(setup.symbol, bars((0, 105, 112, 101, 111)))
        assert setup.status == "ACTIVE"
        assert setup.id not in worker.filled_at
        assert worker.risk_manager.daily_pnl == 0
        assert worker.risk_manager.open_setups == 1

    asyncio.run(database(monkeypatch, run))


def test_intrabar_target_before_possible_fill_is_not_an_invented_win(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)
        await worker.process_paper_bars(setup.symbol, bars((0, 106, 111, 99, 105)))
        assert setup.status == "ACTIVE"
        assert setup.id in worker.filled_at
        await worker.process_paper_bars(setup.symbol, bars((1, 105, 111, 104, 110)))
        assert setup.status == "TP1_HIT"
        assert worker.risk_manager.daily_pnl == pytest.approx(10)
        assert worker.risk_manager.open_setups == 0

    asyncio.run(database(monkeypatch, run))


def test_single_target_market_open_fill_and_round_trip_costs(monkeypatch):
    async def run(sessions):
        worker = tracker()
        worker.settings.paper_fee_bps = 5
        worker.settings.paper_slippage_bps = 2
        setup = await add(worker, sessions)
        await worker.process_paper_bars(setup.symbol, bars((0, 100, 111, 99, 110)))
        assert setup.status == "TP1_HIT"
        assert worker.risk_manager.daily_pnl == pytest.approx(9.86)
        assert setup.id not in worker.active

    asyncio.run(database(monkeypatch, run))


def test_stop_wins_ambiguous_bar_touching_both_levels(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)
        await worker.process_paper_bars(setup.symbol, bars((0, 100, 111, 94, 105)))
        assert setup.status == "SL_HIT"
        assert worker.risk_manager.daily_pnl == pytest.approx(-5)

    asyncio.run(database(monkeypatch, run))


def test_short_one_target_and_limit_fill_use_correct_direction(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = signal()
        setup.direction = "SHORT"
        setup.htf_bias = "BEARISH"
        setup.stop_loss = 105
        setup.take_profits = [90]
        await add(worker, sessions, setup)
        # Target touch cannot fill a sell limit above this entire bar.
        await worker.process_paper_bars(setup.symbol, bars((0, 96, 99, 89, 92)))
        assert setup.id not in worker.filled_at
        # Opening at the sell limit makes subsequent TP reachable after filling.
        await worker.process_paper_bars(setup.symbol, bars((1, 100, 102, 89, 90)))
        assert setup.status == "TP1_HIT"
        assert worker.risk_manager.daily_pnl == pytest.approx(10)

    asyncio.run(database(monkeypatch, run))


def test_bars_before_signal_confirmation_cannot_fill_or_close_limit(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = signal(timestamp=SIGNAL_TIME + timedelta(minutes=1))
        await add(worker, sessions, setup)
        await worker.process_paper_bars(setup.symbol, bars((0, 100, 111, 94, 110)))
        assert setup.status == "ACTIVE"
        assert setup.id not in worker.filled_at
        assert worker.risk_manager.daily_pnl == 0

    asyncio.run(database(monkeypatch, run))


def test_adverse_gap_stop_uses_bar_open_after_prior_fill(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)
        await worker.process_paper_bars(setup.symbol, bars((0, 100, 103, 98, 101)))
        await worker.process_paper_bars(setup.symbol, bars((1, 90, 94, 89, 92)))
        assert setup.status == "SL_HIT"
        assert worker.risk_manager.daily_pnl == pytest.approx(-10)

    asyncio.run(database(monkeypatch, run))


def test_pending_paper_entry_expires_without_realized_pnl(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)
        await worker.process_paper_bars(setup.symbol, bars((31, 105, 106, 101, 105)))
        assert setup.status == "EXPIRED"
        assert worker.risk_manager.daily_pnl == 0
        async with sessions() as session:
            assert (await session.get(SetupRecord, setup.id)).pnl_usdt is None

    asyncio.run(database(monkeypatch, run))


def test_cancel_pending_paper_releases_risk_and_persists_status(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)
        assert await worker.cancel(setup.id) == {"status": "CANCELLED"}
        assert worker.risk_manager.open_setups == 0
        assert setup.id not in worker.active
        async with sessions() as session:
            assert (await session.get(SetupRecord, setup.id)).status == "CANCELLED"

    asyncio.run(database(monkeypatch, run))


def test_cancel_filled_paper_position_is_rejected(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)
        await worker.process_paper_bars(setup.symbol, bars((0, 100, 103, 98, 101)))
        with pytest.raises(ValueError, match="filled"):
            await worker.cancel(setup.id)
        assert setup.id in worker.active
        assert worker.risk_manager.open_setups == 1

    asyncio.run(database(monkeypatch, run))


def test_paper_fill_and_last_processed_bar_survive_restart_without_replay(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)
        entry_bar = bars((0, 106, 111, 99, 105))
        await worker.process_paper_bars(setup.symbol, entry_bar)
        restarted = tracker()
        await restarted.restore()
        assert restarted.filled_at[setup.id] == SIGNAL_TIME
        assert restarted.last_bar[setup.id] == SIGNAL_TIME
        await restarted.process_paper_bars(setup.symbol, entry_bar)
        assert setup.id in restarted.active
        assert restarted.risk_manager.daily_pnl == 0
        await restarted.process_paper_bars(setup.symbol, bars((1, 105, 111, 104, 110)))
        assert restarted.risk_manager.daily_pnl == pytest.approx(10)

    asyncio.run(database(monkeypatch, run))


def test_successful_paper_poll_clears_recoverable_transport_halt(monkeypatch):
    async def run(sessions):
        worker = tracker()
        worker.risk_manager.halted = True
        await worker.poll()
        assert not worker.risk_manager.halted

    asyncio.run(database(monkeypatch, run))


def test_paper_mode_never_manages_live_rows_and_retains_live_halt(monkeypatch):
    async def run(sessions):
        worker = tracker()
        async with sessions() as session:
            await save_setup(session, signal("live-row"), "live")
        await worker.restore()
        assert not worker.active
        assert worker.risk_manager.halted
        await worker.poll()
        assert worker.risk_manager.halted
        worker.executor.reconcile.assert_not_awaited()
        worker.client.get_klines.assert_not_awaited()

    asyncio.run(database(monkeypatch, run))


def test_restart_restores_current_utc_daily_loss_only_for_matching_mode(monkeypatch):
    async def run(sessions):
        today = datetime.now(timezone.utc)
        async with sessions() as session:
            await save_setup(session, signal("today-paper", today), "paper")
            await update_setup_status(session, "today-paper", "SL_HIT", -21)
            await save_setup(session, signal("today-live", today), "live")
            await update_setup_status(session, "today-live", "SL_HIT", -500)
            await save_setup(session, signal("yesterday-paper", today - timedelta(days=1)), "paper")
            await update_setup_status(session, "yesterday-paper", "SL_HIT", -100)
            yesterday = await session.get(SetupRecord, "yesterday-paper")
            yesterday.closed_at = today - timedelta(days=1)
            await session.commit()
        worker = tracker()
        await worker.restore()
        assert worker.risk_manager.daily_pnl == -21
        assert not worker.risk_manager.can_open_setup()
        await worker.restore()
        assert worker.risk_manager.daily_pnl == -21

    asyncio.run(database(monkeypatch, run))


def test_live_signal_without_execution_journal_is_cancelled_on_restore(monkeypatch):
    async def run(sessions):
        worker = tracker(auto=True)
        async with sessions() as session:
            await save_setup(session, signal(), "live")
        await worker.restore()
        assert not worker.active
        assert worker.risk_manager.open_setups == 0
        async with sessions() as session:
            assert (await session.get(SetupRecord, "signal-a")).status == "CANCELLED"

    asyncio.run(database(monkeypatch, run))


def test_unknown_live_journal_restores_halt_and_reservation(monkeypatch):
    async def run(sessions):
        worker = tracker(auto=True)
        setup = signal()
        async with sessions() as session:
            await save_setup(session, setup, "live")
            session.add(ExecutionRecord(setup_id=setup.id, external_oid="external-a", symbol=setup.symbol,
                direction="LONG", state="UNKNOWN", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)))
            await session.commit()
        await worker.restore()
        assert worker.risk_manager.halted
        assert worker.risk_manager.open_setups == 1
        assert setup.id in worker.active

    asyncio.run(database(monkeypatch, run))


def test_live_poll_uses_exchange_confirmation_and_never_paper_price_touches(monkeypatch):
    async def run(sessions):
        worker = tracker(auto=True)
        setup = await add(worker, sessions)
        worker.client.get_klines.return_value = bars((0, 100, 200, 1, 95))
        await worker.poll()
        assert setup.id in worker.active
        assert worker.risk_manager.open_setups == 1
        worker.client.get_klines.assert_not_awaited()
        worker.executor.reconcile.assert_awaited_once_with(setup)

    asyncio.run(database(monkeypatch, run))


def test_live_cancel_waits_for_exchange_confirmation(monkeypatch):
    async def run(sessions):
        worker = tracker(auto=True)
        setup = await add(worker, sessions)
        result = await worker.cancel(setup.id)
        assert result["exchange_confirmation_pending"]
        assert setup.id in worker.active
        assert worker.risk_manager.open_setups == 1
        worker.executor.cancel_pending.assert_awaited_once_with(setup.id)
        worker.executor.reconcile.return_value = ("CANCELLED", setup.entry, 0.0)
        await worker.poll()
        assert setup.id not in worker.active
        assert worker.risk_manager.open_setups == 0

    asyncio.run(database(monkeypatch, run))


def test_close_is_idempotent_and_broadcast_failure_cannot_leave_active_trade(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)
        worker.ws.broadcast.side_effect = RuntimeError("websocket unavailable")
        await asyncio.gather(worker._close_setup(setup, "TP1_HIT", 5), worker._close_setup(setup, "TP1_HIT", 5))
        assert worker.risk_manager.daily_pnl == 5
        assert worker.risk_manager.open_setups == 0
        assert setup.id not in worker.active
        async with sessions() as session:
            record = await session.get(SetupRecord, setup.id)
            assert record.status == "TP1_HIT"
            assert record.pnl_usdt == 5

    asyncio.run(database(monkeypatch, run))


def test_database_close_failure_preserves_risk_and_active_setup(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)

        class BrokenSession:
            async def __aenter__(self):
                raise RuntimeError("database unavailable")

            async def __aexit__(self, *args):
                return False

        monkeypatch.setattr(tracker_module, "SessionLocal", BrokenSession)
        with pytest.raises(RuntimeError):
            await worker._close_setup(setup, "SL_HIT", -5)
        assert worker.risk_manager.daily_pnl == 0
        assert worker.risk_manager.open_setups == 1
        assert setup.id in worker.active

    asyncio.run(database(monkeypatch, run))


def history(start, count):
    return pd.DataFrame({"open": 100.0, "high": 103.0, "low": 98.0, "close": 101.0},
        index=pd.date_range(start, periods=count, freq="min"))


def test_paper_poll_restores_long_outage_and_early_stop_causally(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)
        await worker.process_paper_bars(setup.symbol, bars((0, 100, 103, 98, 101)))
        restarted = tracker()
        await restarted.restore()
        now = SIGNAL_TIME + timedelta(minutes=3001, seconds=2)
        restarted.client.get_server_time = AsyncMock(return_value=int(now.timestamp() * 1000))
        candles = history(SIGNAL_TIME + timedelta(minutes=1), 3000)
        candles.iloc[0] = [90, 94, 89, 92]
        restarted.client.get_klines.return_value = candles
        await restarted.poll()
        assert restarted.active == {}
        assert restarted.risk_manager.daily_pnl == pytest.approx(-10)
        restarted.client.get_klines.assert_awaited_once_with(setup.symbol, "1m", limit=3000,
            start_time=int((SIGNAL_TIME + timedelta(minutes=1)).timestamp() * 1000),
            end_time=int(now.timestamp() * 1000) - 1500)
        async with sessions() as session:
            record = await session.get(SetupRecord, setup.id)
            assert record.status == "SL_HIT"
            assert record.pnl_usdt == pytest.approx(-10)

    asyncio.run(database(monkeypatch, run))


def test_paper_backfill_missing_candle_preserves_cursor_and_halts(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)
        await worker.process_paper_bars(setup.symbol, bars((0, 100, 103, 98, 101)))
        now = SIGNAL_TIME + timedelta(minutes=4, seconds=2)
        worker.client.get_server_time = AsyncMock(return_value=int(now.timestamp() * 1000))
        candles = history(SIGNAL_TIME + timedelta(minutes=1), 3)
        candles.iloc[0] = [90, 94, 89, 92]
        worker.client.get_klines.return_value = candles.drop(candles.index[1])
        with pytest.raises(ValueError, match="incomplete"):
            await worker.poll()
        assert worker.risk_manager.halted
        assert worker.last_bar[setup.id] == SIGNAL_TIME
        assert worker.risk_manager.daily_pnl == 0
        assert worker.risk_manager.open_setups == 1
        async with sessions() as session:
            record = await session.get(SetupRecord, setup.id)
            assert record.status == "ACTIVE"
            assert record.paper_last_bar_at.replace(tzinfo=timezone.utc) == SIGNAL_TIME
        worker.client.get_klines.return_value = candles
        await worker.poll()
        assert not worker.risk_manager.halted
        assert worker.risk_manager.daily_pnl == pytest.approx(-10)

    asyncio.run(database(monkeypatch, run))


def test_paper_backfill_too_old_halts_without_truncating_history(monkeypatch):
    async def run(sessions):
        worker = tracker()
        setup = await add(worker, sessions)
        worker.PAPER_BACKFILL_LIMIT = 3
        now = SIGNAL_TIME + timedelta(minutes=6, seconds=2)
        worker.client.get_server_time = AsyncMock(return_value=int(now.timestamp() * 1000))
        with pytest.raises(ValueError, match="recoverable"):
            await worker.poll()
        assert worker.risk_manager.halted
        assert setup.id in worker.active
        assert setup.id not in worker.last_bar
        worker.client.get_klines.assert_not_awaited()

    asyncio.run(database(monkeypatch, run))


@pytest.mark.parametrize("today_exit,expected_status,expected_pnl,blocked", [
    ([100, 101, 94, 96], "SL_HIT", -25, True),
    ([105, 111, 104, 110], "TP1_HIT", 50, False),
])
def test_restart_backfill_assigns_losses_to_actual_utc_close_day(
    monkeypatch, today_exit, expected_status, expected_pnl, blocked
):
    now = datetime(2026, 10, 4, 0, 10, tzinfo=timezone.utc)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz else now.replace(tzinfo=None)

    monkeypatch.setattr(tracker_module, "datetime", FrozenDateTime)
    monkeypatch.setattr(risk_module, "datetime", FrozenDateTime)

    async def run(sessions):
        midnight = now.replace(hour=0, minute=0)
        old_signal_at = midnight - timedelta(minutes=5)
        before_restart = tracker()
        yesterday_setup = signal("yesterday-stop", old_signal_at)
        yesterday_setup.position_size_usdt = 600
        await add(before_restart, sessions, yesterday_setup)
        # A filled paper position and durable cursor survive midnight and restart.
        await before_restart.process_paper_bars(yesterday_setup.symbol, history(old_signal_at, 1))
        restarted = tracker()
        await restarted.restore()
        assert restarted.risk_manager.daily_opening_balance_source == "restored_paper_ledger"
        restarted.client.get_server_time = AsyncMock(return_value=int(now.timestamp() * 1000))
        old_candles = history(old_signal_at + timedelta(minutes=1), 13)
        old_stop_at = midnight - timedelta(minutes=2)
        old_candles.loc[old_stop_at] = [100, 101, 94, 96]
        restarted.client.get_klines.return_value = old_candles
        await restarted.poll()
        assert restarted.risk_manager.balance == pytest.approx(970)
        assert restarted.risk_manager.daily_pnl == 0
        assert restarted.risk_manager.daily_opening_balance == pytest.approx(970)
        assert restarted.risk_manager.can_open_setup()
        async with sessions() as session:
            old_record = await session.get(SetupRecord, yesterday_setup.id)
            assert old_record.status == "SL_HIT"
            assert old_record.pnl_usdt == pytest.approx(-30)
            assert old_record.closed_at.replace(tzinfo=timezone.utc) == midnight - timedelta(minutes=1)

        today_setup = signal("today-exit", midnight)
        today_setup.symbol = "ETH_USDT"
        today_setup.position_size_usdt = 500
        await add(restarted, sessions, today_setup)
        today_candles = history(midnight, 9)
        today_candles.loc[midnight + timedelta(minutes=2)] = today_exit
        restarted.client.get_klines.return_value = today_candles
        await restarted.poll()
        assert restarted.risk_manager.daily_pnl == pytest.approx(expected_pnl)
        assert restarted.risk_manager.balance == pytest.approx(970 + expected_pnl)
        assert restarted.risk_manager.daily_opening_balance == pytest.approx(970)
        assert restarted.risk_manager.can_open_setup() is not blocked
        async with sessions() as session:
            today_record = await session.get(SetupRecord, today_setup.id)
            assert today_record.status == expected_status
            assert today_record.closed_at.replace(tzinfo=timezone.utc) == midnight + timedelta(minutes=3)
        restored_again = tracker()
        await restored_again.restore()
        assert restored_again.risk_manager.daily_pnl == pytest.approx(expected_pnl)
        assert restored_again.risk_manager.balance == pytest.approx(970 + expected_pnl)
        assert restored_again.risk_manager.daily_opening_balance == pytest.approx(970)
        assert restored_again.risk_manager.can_open_setup() is not blocked

    asyncio.run(database(monkeypatch, run))
