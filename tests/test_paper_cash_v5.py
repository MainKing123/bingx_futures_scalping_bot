"""Meaningful durable paper wallet tests; no public/private network calls."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import Base, SetupRecord
from app.db.repository import save_setup
from app.execution.economics import RuntimeTradeSetup
from app.risk import risk_manager as risk_module
from app.risk.risk_manager import RiskManager
from app.runtime_settings import RuntimeSettings
from app.tracking import trade_tracker as tracker_module
from app.tracking.trade_tracker import TradeTracker


START = datetime(2026, 10, 5, 0, 0, tzinfo=timezone.utc)


def idea(identifier="cash-v5", direction="LONG", timestamp=START):
    return RuntimeTradeSetup(id=identifier, timestamp=timestamp, symbol="BTC_USDT", direction=direction,
        setup_type="VOLIUM_INTRADAY", htf_bias="BULLISH" if direction == "LONG" else "BEARISH",
        entry=100, stop_loss=95 if direction == "LONG" else 105,
        take_profits=[110 if direction == "LONG" else 90], risk_reward=2,
        position_size_usdt=100, leverage=50,
        economics={"modeled_fee_bps_per_side": 5, "modeled_slippage_bps_per_side": 2,
            "reserved_margin_usdt": 2.07, "initial_margin_usdt": 2})


def candles(*rows, start=START):
    return pd.DataFrame([dict(open=o, high=h, low=low, close=c) for _, o, h, low, c in rows],
        index=pd.DatetimeIndex([start + timedelta(minutes=minute) for minute, *_ in rows]))


def funding_client(event=START - timedelta(minutes=30), rate=.001, *, observed=START + timedelta(minutes=10)):
    history = [{"symbol": "BTC_USDT", "settleTime": int((event - timedelta(hours=i)).timestamp()*1000),
        "fundingRate": rate, "collectCycle": 1} for i in range(3)]
    current = {"symbol": "BTC_USDT", "collectCycle": 1,
        "nextSettleTime": int((event + timedelta(hours=1)).timestamp()*1000),
        "timestamp": int(observed.timestamp()*1000)}

    async def request(method, path, params=None, **kwargs):
        assert method == "GET" and kwargs == {"signed": False}
        if path.endswith("/history"):
            return {"resultList": history}
        assert path == "/api/v1/contract/funding_rate/BTC_USDT"
        return current

    async def rates(symbol, since_ms):
        records = sorted([record for record in history if record["settleTime"] >= since_ms], key=lambda r: r["settleTime"])
        return pd.Series([record["fundingRate"] for record in records],
            index=pd.to_datetime([record["settleTime"] for record in records], unit="ms", utc=True), dtype=float)

    client = SimpleNamespace(get_funding_history=AsyncMock(side_effect=rates),
        _request=AsyncMock(side_effect=request), history=history, current=current,
        get_klines=AsyncMock(), get_server_time=AsyncMock(), get_balance=AsyncMock())
    return client


def worker(client=None):
    config = RuntimeSettings(_env_file=None, pair_selection="fixed")
    return TradeTracker(client or funding_client(), config, RiskManager(config), SimpleNamespace())


def isolated(monkeypatch, action, *, now=START + timedelta(minutes=10)):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz else now.replace(tzinfo=None)

    monkeypatch.setattr(tracker_module, "datetime", Clock)
    monkeypatch.setattr(risk_module, "datetime", Clock)

    async def run():
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(tracker_module, "SessionLocal", sessions)
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        try:
            await action(sessions)
        finally:
            await engine.dispose()
    asyncio.run(run())


async def persist(sessions, tracker, setup=None):
    setup = setup or idea()
    async with sessions() as session:
        await save_setup(session, setup, "paper")
    await tracker.add(setup)
    tracker.risk_manager.open_setups += 1
    return setup


def test_entry_cost_is_cash_at_fill_and_exit_uses_actual_exit_notional(monkeypatch):
    async def action(sessions):
        tracker = worker()
        setup = await persist(sessions, tracker)
        await tracker.process_paper_bars(setup.symbol, candles((0, 100, 103, 99, 102)))
        assert tracker.risk_manager.balance == pytest.approx(999.93)
        assert tracker.risk_manager.daily_pnl == pytest.approx(-.07)
        assert setup.economics["reserved_margin_usdt"] == 2  # Fee was paid, no second reservation.
        assert tracker.marked_equity == pytest.approx(1001.93)
        await tracker.process_paper_bars(setup.symbol, candles((1, 102, 111, 101, 110)))
        assert tracker.risk_manager.balance == pytest.approx(1009.853)
        assert tracker.risk_manager.daily_pnl == pytest.approx(9.853)
        async with sessions() as session:
            row = await session.get(SetupRecord, setup.id)
            assert row.pnl_usdt == pytest.approx(9.853)
            events = json.loads(row.payload)["economics"]["paper_ledger"]["events"]
            assert events[-1]["fee_usdt"] == pytest.approx(.055)
            assert events[-1]["slippage_usdt"] == pytest.approx(.022)
    isolated(monkeypatch, action)


def test_restart_twice_does_not_rebook_entry_funding_or_exit(monkeypatch):
    async def action(sessions):
        client = funding_client(START + timedelta(minutes=1))
        tracker = worker(client)
        setup = await persist(sessions, tracker)
        first = candles((0, 100, 103, 99, 102))
        await tracker.process_paper_bars(setup.symbol, first)
        assert tracker.risk_manager.balance == pytest.approx(999.828)
        restarted = worker(client)
        await restarted.restore()
        await restarted.restore()
        assert restarted.risk_manager.balance == pytest.approx(999.828)
        assert restarted.marked_equity == pytest.approx(1001.828)
        await restarted.process_paper_bars(setup.symbol, first)
        assert restarted.risk_manager.balance == pytest.approx(999.828)
        await restarted.process_paper_bars(setup.symbol, candles((1, 102, 111, 101, 110)))
        assert restarted.risk_manager.balance == pytest.approx(1009.751)
        await restarted.restore()
        await restarted.restore()
        assert restarted.risk_manager.balance == pytest.approx(1009.751)
        assert restarted.risk_manager.daily_pnl == pytest.approx(9.751)
        async with sessions() as session:
            rec = await session.get(SetupRecord, setup.id)
            events = json.loads(rec.payload)["economics"]["paper_ledger"]["events"]
            assert len([event for event in events if event["id"].startswith("funding:")]) == 1
            assert rec.pnl_usdt == pytest.approx(9.751)
    isolated(monkeypatch, action)


@pytest.mark.parametrize("rate,expected,skipped", [(.001, -.106, False), (-.001, 0, True)])
def test_fill_at_settlement_unknown_intrabar_holding_is_conservative(monkeypatch, rate, expected, skipped):
    async def action(sessions):
        tracker = worker(funding_client(START, rate))
        setup = await persist(sessions, tracker)
        await tracker.process_paper_bars(setup.symbol, candles((0, 106, 107, 99, 103)))
        assert tracker.risk_manager.balance == pytest.approx(999.93 + expected)
        event = setup.economics["paper_ledger"]["events"][1]
        assert event["cash_delta_usdt"] == pytest.approx(expected)
        assert event["favorable_skipped_intrabar_uncertainty"] == skipped
    isolated(monkeypatch, action)


@pytest.mark.parametrize("rate,expected", [(.001, -.102), (-.001, .102)])
def test_funding_when_known_held_uses_actual_rate_and_observed_closed_price(monkeypatch, rate, expected):
    async def action(sessions):
        tracker = worker(funding_client(START + timedelta(minutes=1), rate))
        setup = await persist(sessions, tracker)
        await tracker.process_paper_bars(setup.symbol, candles((0, 100, 103, 99, 102)))
        assert tracker.risk_manager.balance == pytest.approx(999.93 + expected)
        assert setup.economics["paper_ledger"]["events"][1]["observed_price_proxy"] == 102
    isolated(monkeypatch, action)


def test_gap_loss_is_not_clipped_and_exit_fee_follows_gap_notional(monkeypatch):
    async def action(sessions):
        tracker = worker()
        setup = await persist(sessions, tracker)
        await tracker.process_paper_bars(setup.symbol, candles((0, 100, 103, 99, 102)))
        await tracker.process_paper_bars(setup.symbol, candles((1, 90, 94, 89, 92)))
        assert tracker.risk_manager.balance == pytest.approx(989.867)
        async with sessions() as session:
            rec = await session.get(SetupRecord, setup.id)
            assert rec.pnl_usdt == pytest.approx(-10.133)
            event = json.loads(rec.payload)["economics"]["paper_ledger"]["events"][-1]
            assert event["exit_price"] == 90
            assert event["fee_usdt"] == pytest.approx(.045)
    isolated(monkeypatch, action)


@pytest.mark.parametrize("failure", ["missing_event", "missing_cycle", "missing_rates", "stale_schedule", "transport"])
def test_missing_funding_fails_closed_before_fill_or_cursor_commit(monkeypatch, failure):
    async def action(sessions):
        client = funding_client(START)
        if failure == "missing_event":
            client.history.pop(1)
        elif failure == "missing_cycle":
            client.history[0].pop("collectCycle")
        elif failure == "missing_rates":
            client.get_funding_history.side_effect = None
            client.get_funding_history.return_value = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        elif failure == "stale_schedule":
            client.current["timestamp"] -= 100_000
        else:
            client.get_funding_history.side_effect = RuntimeError("public API unavailable")
        tracker = worker(client)
        setup = await persist(sessions, tracker)
        with pytest.raises((ValueError, RuntimeError)):
            await tracker.process_paper_bars(setup.symbol, candles((0, 100, 103, 99, 102)))
        assert tracker.risk_manager.halted
        assert tracker.risk_manager.balance == 1000
        assert setup.id not in tracker.filled_at
        async with sessions() as session:
            rec = await session.get(SetupRecord, setup.id)
            assert rec.paper_last_bar_at is None and rec.paper_filled_at is None
    isolated(monkeypatch, action)


def test_failed_database_commit_does_not_change_in_memory_cash(monkeypatch):
    async def action(sessions):
        tracker = worker()
        setup = await persist(sessions, tracker)
        class Broken:
            async def __aenter__(self):
                raise RuntimeError("DB unavailable")
            async def __aexit__(self, *args):
                return False
        monkeypatch.setattr(tracker_module, "SessionLocal", Broken)
        with pytest.raises(RuntimeError, match="DB unavailable"):
            await tracker.process_paper_bars(setup.symbol, candles((0, 100, 103, 99, 102)))
        assert tracker.risk_manager.balance == 1000
        assert setup.id not in tracker.filled_at
        assert "paper_ledger" not in setup.economics
    isolated(monkeypatch, action)


def test_invalid_ohlc_cannot_book_cash(monkeypatch):
    async def action(sessions):
        tracker = worker()
        setup = await persist(sessions, tracker)
        await tracker.process_paper_bars(setup.symbol, candles((0, 100, 103, 99, 98.9)))
    # Invalid OHLC is itself rejected before cash is booked.
    with pytest.raises(ValueError, match="OHLC"):
        isolated(monkeypatch, action)


def test_missing_candle_does_not_advance_cash_or_funding_cursor(monkeypatch):
    async def action(sessions):
        tracker = worker()
        setup = await persist(sessions, tracker)
        await tracker.process_paper_bars(setup.symbol, candles((0, 100, 103, 99, 102)))
        with pytest.raises(ValueError, match="history is incomplete"):
            await tracker.process_paper_bars(setup.symbol, candles((2, 102, 111, 101, 110)))
        assert tracker.risk_manager.balance == pytest.approx(999.93)
        assert tracker.last_bar[setup.id] == START
        assert tracker.risk_manager.halted
    isolated(monkeypatch, action)


def test_missing_mark_halts_and_unrealized_loss_reduces_admission_equity(monkeypatch):
    async def action(sessions):
        tracker = worker()
        setup = await persist(sessions, tracker)
        await tracker.process_paper_bars(setup.symbol, candles((0, 100, 101, 97, 98)))
        assert tracker.marked_equity == pytest.approx(997.93)
        tracker._marks.pop(setup.id)
        with pytest.raises(ValueError, match="no observed price"):
            _ = tracker.marked_equity
        assert tracker.risk_manager.halted
    isolated(monkeypatch, action)


def test_missing_first_bar_cannot_invent_a_late_pending_fill(monkeypatch):
    async def action(sessions):
        tracker = worker()
        setup = await persist(sessions, tracker)
        with pytest.raises(ValueError, match="does not begin"):
            await tracker.process_paper_bars(setup.symbol, candles((2, 100, 103, 99, 102)))
        assert tracker.risk_manager.balance == 1000
    isolated(monkeypatch, action)


def test_intrabar_target_before_possible_fill_remains_unproved(monkeypatch):
    async def action(sessions):
        tracker = worker()
        setup = await persist(sessions, tracker)
        await tracker.process_paper_bars(setup.symbol, candles((0, 106, 111, 99, 105)))
        assert setup.status == "ACTIVE"
        assert tracker.risk_manager.balance == pytest.approx(999.93)
        await tracker.process_paper_bars(setup.symbol, candles((1, 105, 111, 104, 110)))
        assert tracker.risk_manager.balance == pytest.approx(1009.853)
    isolated(monkeypatch, action)


def test_daily_pnl_restores_cash_event_dates_across_midnight(monkeypatch):
    async def action(sessions):
        tracker = worker(funding_client(START))
        setup = await persist(sessions, tracker, idea(timestamp=START - timedelta(minutes=1)))
        await tracker.process_paper_bars(setup.symbol, candles((0, 100, 103, 99, 102), start=START-timedelta(minutes=1)))
        assert tracker.risk_manager.balance == pytest.approx(999.828)
        assert tracker.risk_manager.daily_pnl == pytest.approx(-.102)
        restarted = worker(tracker.client)
        await restarted.restore()
        await restarted.restore()
        assert restarted.risk_manager.daily_pnl == pytest.approx(-.102)
        assert restarted.risk_manager.daily_opening_balance == pytest.approx(999.93)
    isolated(monkeypatch, action)


def test_filled_runtime_without_durable_entry_ledger_halts_restore(monkeypatch):
    async def action(sessions):
        tracker = worker()
        setup = await persist(sessions, tracker)
        async with sessions() as session:
            rec = await session.get(SetupRecord, setup.id)
            rec.paper_filled_at = START
            await session.commit()
        with pytest.raises(ValueError, match="no durable cash ledger"):
            await tracker.restore()
        assert tracker.risk_manager.halted
    isolated(monkeypatch, action)


@pytest.mark.parametrize("rate,expected", [(.001, -.102), (-.001, 0)])
def test_exit_during_settlement_bar_charges_adverse_and_skips_unproved_favorable(monkeypatch, rate, expected):
    async def action(sessions):
        client = funding_client(START + timedelta(minutes=1, seconds=30), rate)
        tracker = worker(client)
        setup = await persist(sessions, tracker)
        await tracker.process_paper_bars(setup.symbol, candles((0, 100, 103, 99, 102)))
        await tracker.process_paper_bars(setup.symbol, candles((1, 102, 111, 101, 110)))
        assert tracker.risk_manager.balance == pytest.approx(1009.853 + expected)
    isolated(monkeypatch, action)


def test_short_exit_cost_uses_lower_exit_notional(monkeypatch):
    async def action(sessions):
        tracker = worker()
        setup = await persist(sessions, tracker, idea(direction="SHORT"))
        await tracker.process_paper_bars(setup.symbol, candles((0, 100, 102, 89, 90)))
        assert tracker.risk_manager.balance == pytest.approx(1009.867)
        assert tracker.risk_manager.daily_pnl == pytest.approx(9.867)
    isolated(monkeypatch, action)


def test_stop_wins_both_touches_and_full_frame_gap_is_validated_before_cash(monkeypatch):
    async def action(sessions):
        tracker = worker()
        setup = await persist(sessions, tracker)
        with pytest.raises(ValueError, match="incomplete"):
            await tracker.process_paper_bars(setup.symbol, candles((0, 100, 103, 99, 102), (2, 102, 111, 101, 110)))
        assert tracker.risk_manager.balance == 1000
        assert setup.id not in tracker.last_bar
        await tracker.process_paper_bars(setup.symbol, candles((0, 100, 111, 94, 105)))
        assert setup.status == "SL_HIT"
        assert tracker.risk_manager.balance == pytest.approx(994.8635)
    isolated(monkeypatch, action)


def test_restore_requires_successful_public_reconciliation_even_if_no_new_candle(monkeypatch):
    now = START + timedelta(minutes=1, seconds=2)
    async def action(sessions):
        client = funding_client(observed=now)
        tracker = worker(client)
        setup = await persist(sessions, tracker)
        await tracker.process_paper_bars(setup.symbol, candles((0, 100, 103, 99, 102)))
        restarted = worker(client)
        await restarted.restore()
        assert restarted.risk_manager.halted
        client.get_server_time.return_value = int(now.timestamp()*1000)
        client.get_funding_history.side_effect = RuntimeError("funding unavailable")
        with pytest.raises(RuntimeError, match="funding unavailable"):
            await restarted.poll()
        assert restarted.risk_manager.halted
        # Restore a real mocked rate reader after proving the failure cannot clear the halt.
        replacement = funding_client(observed=now)
        restarted.client = replacement
        replacement.get_server_time.return_value = int(now.timestamp()*1000)
        await restarted.poll()
        assert not restarted.risk_manager.halted
        replacement.get_klines.assert_not_awaited()
        assert restarted.risk_manager.balance == pytest.approx(999.93)
    isolated(monkeypatch, action, now=now)


def test_cancel_rechecks_fill_after_a_concurrent_cash_commit(monkeypatch):
    async def action(sessions):
        client = funding_client()
        tracker = worker(client)
        setup = await persist(sessions, tracker)
        entered, release = asyncio.Event(), asyncio.Event()
        original = client.get_funding_history.side_effect
        async def paused_rates(symbol, since):
            entered.set()
            await release.wait()
            return await original(symbol, since)
        client.get_funding_history.side_effect = paused_rates
        process = asyncio.create_task(tracker.process_paper_bars(setup.symbol, candles((0, 100, 103, 99, 102))))
        await entered.wait()
        cancel = asyncio.create_task(tracker.cancel(setup.id))
        release.set()
        await process
        with pytest.raises(ValueError, match="filled"):
            await cancel
        assert setup.status == "ACTIVE"
        assert tracker.risk_manager.balance == pytest.approx(999.93)
    isolated(monkeypatch, action)
