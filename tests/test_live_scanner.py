import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.models import Base, SetupRecord
from app.risk.risk_manager import RiskManager
from app.scanner import scanner as scanner_module
from app.scanner.scanner import SignalScanner
from app.schemas.setup import TradeSetup


SIGNAL_TIME = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)


def make_setup(symbol="BTC_USDT", direction="LONG", timestamp=SIGNAL_TIME):
    return TradeSetup(
        id=hashlib.sha256(f"{symbol}|{direction}|{timestamp.isoformat()}".encode()).hexdigest()[:24],
        timestamp=timestamp,
        symbol=symbol,
        direction=direction,
        setup_type="VOLIUM_INTRADAY",
        htf_bias="BULLISH" if direction == "LONG" else "BEARISH",
        entry=100,
        stop_loss=95 if direction == "LONG" else 105,
        take_profits=[110, 115, 120] if direction == "LONG" else [90, 85, 80],
        risk_reward=2,
        confidence="HIGH",
        confluences=["context sweep", "entry engulfing"],
        position_size_usdt=200,
    )


def make_scanner(*, auto_execution=False, max_open=5, mode="intraday", executor=None):
    settings = Settings(_env_file=None, pair_selection="fixed", auto_execution=auto_execution, max_open_setups=max_open, volium_mode=mode)
    client = SimpleNamespace(get_klines=AsyncMock())
    tracker = SimpleNamespace(add=AsyncMock())
    ws = SimpleNamespace(broadcast=AsyncMock())
    risk = RiskManager(settings)
    return SignalScanner(client, settings, risk, tracker, executor, ws)


async def with_database(monkeypatch, action):
    # Isolated database and mocked exchange only; these tests cannot submit orders.
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(scanner_module, "SessionLocal", sessions)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        await action(sessions)
    finally:
        await engine.dispose()


def frame(timestamp=SIGNAL_TIME):
    return pd.DataFrame(
        {"open": [100], "high": [105], "low": [95], "close": [101], "volume": [10]},
        index=pd.DatetimeIndex([timestamp]),
    )


def test_persisted_signal_id_blocks_replay_even_after_closure(monkeypatch):
    async def run(sessions):
        scanner = make_scanner()
        first = make_setup()
        assert await scanner._publish_setup(first)
        async with sessions() as session:
            record = await session.get(SetupRecord, first.id)
            record.status = "CANCELLED"
            await session.commit()
        scanner.risk_manager.open_setups = 0
        assert not await scanner._publish_setup(make_setup())
        scanner.tracker.add.assert_awaited_once()

    asyncio.run(with_database(monkeypatch, run))


@pytest.mark.parametrize("mode", ["intraday", "scalp", "swing"])
def test_concurrent_publications_respect_open_limit_in_every_mode(monkeypatch, mode):
    async def run(sessions):
        scanner = make_scanner(max_open=1, mode=mode)
        results = await asyncio.gather(
            scanner._publish_setup(make_setup("BTC_USDT")),
            scanner._publish_setup(make_setup("ETH_USDT")),
        )
        assert sorted(results) == [False, True]
        assert scanner.risk_manager.open_setups == 1
        async with sessions() as session:
            assert len((await session.scalars(select(SetupRecord))).all()) == 1

    asyncio.run(with_database(monkeypatch, run))


def test_persisted_active_symbol_blocks_new_signal_after_restart(monkeypatch):
    async def run(sessions):
        scanner = make_scanner()
        assert await scanner._publish_setup(make_setup())
        restarted = make_scanner()
        opposite_signal = make_setup(direction="SHORT", timestamp=SIGNAL_TIME + timedelta(minutes=5))
        assert not await restarted._publish_setup(opposite_signal)
        restarted.tracker.add.assert_not_awaited()

    asyncio.run(with_database(monkeypatch, run))


def test_live_execution_receives_persisted_signal_and_reserved_risk(monkeypatch):
    async def run(sessions):
        scanner = make_scanner(auto_execution=True)

        async def execute(setup):
            assert scanner.risk_manager.open_setups == 1
            async with sessions() as session:
                assert await session.get(SetupRecord, setup.id) is not None
            return True

        scanner.executor = SimpleNamespace(execute_setup=AsyncMock(side_effect=execute))
        assert await scanner._publish_setup(make_setup())
        scanner.executor.execute_setup.assert_awaited_once()
        scanner.tracker.add.assert_awaited_once()
        assert scanner.risk_manager.open_setups == 1

    asyncio.run(with_database(monkeypatch, run))


def test_definitive_exchange_rejection_releases_reservation(monkeypatch):
    async def run(sessions):
        executor = SimpleNamespace(execute_setup=AsyncMock(return_value=False))
        scanner = make_scanner(auto_execution=True, executor=executor)
        setup = make_setup()
        assert not await scanner._publish_setup(setup)
        assert scanner.risk_manager.open_setups == 0
        scanner.tracker.add.assert_not_awaited()
        async with sessions() as session:
            assert (await session.get(SetupRecord, setup.id)).status == "CANCELLED"

    asyncio.run(with_database(monkeypatch, run))


def test_execution_interruption_keeps_persisted_reservation_and_tracker(monkeypatch):
    async def run(sessions):
        executor = SimpleNamespace(execute_setup=AsyncMock(side_effect=TimeoutError("unknown entry outcome")))
        scanner = make_scanner(auto_execution=True, executor=executor)
        setup = make_setup()
        with pytest.raises(TimeoutError):
            await scanner._publish_setup(setup)
        assert scanner.risk_manager.open_setups == 1
        scanner.tracker.add.assert_awaited_once_with(setup)
        async with sessions() as session:
            assert (await session.get(SetupRecord, setup.id)).status == "ACTIVE"

    asyncio.run(with_database(monkeypatch, run))


def test_failed_signal_save_releases_reservation(monkeypatch):
    async def run(sessions):
        monkeypatch.setattr(scanner_module, "save_setup", AsyncMock(side_effect=RuntimeError("database unavailable")))
        scanner = make_scanner()
        with pytest.raises(RuntimeError):
            await scanner._publish_setup(make_setup())
        assert scanner.risk_manager.open_setups == 0
        scanner.tracker.add.assert_not_awaited()

    asyncio.run(with_database(monkeypatch, run))


@pytest.mark.parametrize(
    "mode,swing_context,timeframes",
    [("intraday", "1d", {"1d", "1h", "5m"}), ("scalp", "1d", {"1h", "5m", "1m"}),
     ("swing", "1d", {"1d", "1h"}), ("swing", "1w", {"1w", "4h"})],
)
def test_polling_uses_video_mode_frames_and_skips_processed_candle(monkeypatch, mode, swing_context, timeframes):
    async def run():
        scanner = make_scanner(mode=mode)
        scanner.settings.volium_swing_context = swing_context
        scanner.settings.trading_symbols = ["BTC_USDT"]
        scanner.client.get_klines.return_value = frame()
        analyze = Mock(return_value=None)
        monkeypatch.setattr(scanner_module, "analyze_volium_from_df", analyze)
        await scanner.scan()
        args = analyze.call_args.kwargs
        assert args["symbol"] == "BTC_USDT"
        assert args["mode"] == mode
        assert set(args["frames"]) == timeframes
        assert args["now"].tzinfo == timezone.utc
        assert scanner.last_processed[("BTC_USDT", mode)] == SIGNAL_TIME
        assert all(call.kwargs["limit"] == scanner.settings.volium_context_lookback + 30 for call in scanner.client.get_klines.await_args_list)
        await scanner.scan()
        assert analyze.call_count == 1
        assert scanner.client.get_klines.await_count == len(timeframes) + 1

    asyncio.run(run())


def test_fetch_failure_retries_next_poll_without_marking_candle(monkeypatch):
    async def run():
        scanner = make_scanner()
        scanner.settings.trading_symbols = ["BTC_USDT"]
        scanner.client.get_klines.side_effect = TimeoutError("public candle request timed out")
        analyze = Mock(return_value=None)
        monkeypatch.setattr(scanner_module, "analyze_volium_from_df", analyze)
        await scanner.scan()
        assert ("BTC_USDT", "intraday") not in scanner.last_processed
        assert "BTC_USDT" in scanner.last_errors
        scanner.client.get_klines.side_effect = None
        scanner.client.get_klines.return_value = frame()
        await scanner.scan()
        assert analyze.call_count == 1
        assert "BTC_USDT" not in scanner.last_errors

    asyncio.run(run())


def test_broadcast_failure_does_not_prevent_execution_or_tracking(monkeypatch):
    async def run(sessions):
        executor = SimpleNamespace(execute_setup=AsyncMock(return_value=True))
        scanner = make_scanner(auto_execution=True, executor=executor)
        scanner.ws_manager.broadcast.side_effect = RuntimeError("websocket unavailable")
        assert await scanner._publish_setup(make_setup())
        scanner.executor.execute_setup.assert_awaited_once()
        scanner.tracker.add.assert_awaited_once()

    asyncio.run(with_database(monkeypatch, run))


def test_paper_signal_without_size_gets_persisted_equity_risk_size(monkeypatch):
    async def run(sessions):
        scanner = make_scanner()
        setup = make_setup()
        setup.position_size_usdt = None
        scanner.risk_manager.balance = 2000
        assert await scanner._publish_setup(setup)
        assert setup.position_size_usdt == 200
        async with sessions() as session:
            record = await session.get(SetupRecord, setup.id)
            persisted = TradeSetup.model_validate_json(record.payload)
            assert persisted.position_size_usdt == 200
            assert record.execution_mode == "paper"

    asyncio.run(with_database(monkeypatch, run))


def test_concurrent_paper_signals_share_margin_and_reject_zero_size(monkeypatch):
    async def run(sessions):
        scanner = make_scanner()
        first, second = make_setup("BTC_USDT"), make_setup("ETH_USDT")
        first.stop_loss = second.stop_loss = 99.8
        assert await asyncio.gather(scanner._publish_setup(first), scanner._publish_setup(second)) == [True, True]
        # Each asks for 2500 USDT, but the account has only 2700 in total capacity.
        assert sorted([first.position_size_usdt, second.position_size_usdt]) == pytest.approx([200, 2500])
        third = make_setup("ZEC_USDT")
        third.stop_loss = 99.8
        assert not await scanner._publish_setup(third)
        assert scanner.risk_manager.open_setups == 2
        async with sessions() as session:
            rows = (await session.scalars(select(SetupRecord))).all()
            assert len(rows) == 2
            assert sum(TradeSetup.model_validate_json(row.payload).position_size_usdt for row in rows) == pytest.approx(2700)

    asyncio.run(with_database(monkeypatch, run))


def test_restart_paper_pending_reservation_reduces_available_margin(monkeypatch):
    async def run(sessions):
        first_scanner = make_scanner()
        first = make_setup("BTC_USDT")
        first.stop_loss = 99.8
        assert await first_scanner._publish_setup(first)
        # A new scanner has an empty in-memory tracker. The unfilled DB row still reserves margin.
        restarted = make_scanner()
        second = make_setup("ETH_USDT")
        second.stop_loss = 99.8
        assert await restarted._publish_setup(second)
        assert second.position_size_usdt == pytest.approx(200)
        async with sessions() as session:
            record = await session.get(SetupRecord, first.id)
            assert record.paper_filled_at is None

    asyncio.run(with_database(monkeypatch, run))


def test_missing_paper_reservation_size_halts_instead_of_reusing_margin(monkeypatch):
    async def run(sessions):
        scanner = make_scanner()
        first = make_setup("BTC_USDT")
        assert await scanner._publish_setup(first)
        async with sessions() as session:
            record = await session.get(SetupRecord, first.id)
            first.position_size_usdt = None
            record.payload = first.model_dump_json()
            await session.commit()
        with pytest.raises(ValueError, match="reservation"):
            await scanner._publish_setup(make_setup("ETH_USDT"))
        assert scanner.risk_manager.halted
        assert scanner.risk_manager.open_setups == 1

    asyncio.run(with_database(monkeypatch, run))
