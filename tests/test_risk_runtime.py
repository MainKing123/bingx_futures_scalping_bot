import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import main
from app.config import Settings
from app.db.models import Base, SetupRecord
from app.risk.risk_manager import RiskManager
from app.schemas.setup import TradeSetup
from app.tracking import trade_tracker


def test_live_startup_uses_actual_small_equity_and_restored_today_loss(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(timezone.utc)

    async def init():
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            for identifier, mode, pnl, closed_at in (
                ("today-live", "live", -2, now),
                ("yesterday-live", "live", -7, now - timedelta(days=1)),
                ("today-paper", "paper", -30, now),
            ):
                setup = TradeSetup(id=identifier, timestamp=closed_at, symbol="ETH_USDT", direction="LONG",
                    setup_type="VOLIUM_INTRADAY", htf_bias="BULLISH", entry=100, stop_loss=95,
                    take_profits=[110], risk_reward=2, position_size_usdt=100)
                session.add(SetupRecord(id=identifier, created_at=closed_at, symbol=setup.symbol,
                    status="SL_HIT", execution_mode=mode, payload=setup.model_dump_json(),
                    closed_at=closed_at, pnl_usdt=pnl))
            await session.commit()

    monkeypatch.setattr(main, "init_db", init)
    monkeypatch.setattr(trade_tracker, "SessionLocal", sessions)
    settings = Settings(_env_file=None, auto_execution=True, pair_selection="fixed",
        mexc_api_key="mock-key", mexc_api_secret="mock-secret", account_balance_usdt=1000,
        daily_loss_limit_percent=2)
    exchange = SimpleNamespace(get_balance=AsyncMock(return_value={"equity": 48}), close=AsyncMock())
    app = main.create_app(settings, exchange, start_background=False)
    try:
        with TestClient(app):
            risk = app.state.risk_manager
            assert risk.daily_pnl == -2
            assert risk.balance == 48
            assert risk.daily_opening_balance == 50
            assert risk.daily_opening_balance_source == "exchange_equity_minus_today_bot_pnl_proxy"
            assert not risk.can_open_setup()
            # The obsolete 1000-USDT paper baseline would incorrectly permit this loss.
            assert risk.daily_pnl > -settings.account_balance_usdt * settings.daily_loss_limit_percent / 100
        exchange.close.assert_awaited_once()
    finally:
        asyncio.run(engine.dispose())


@pytest.mark.parametrize("equity", [0, -1, float("nan"), float("inf")])
def test_nonpositive_or_invalid_equity_blocks_entries(equity):
    risk = RiskManager(Settings(_env_file=None))
    risk.balance = equity
    assert not risk.can_open_setup()


def test_live_proxy_reconstruction_and_daily_loss_boundary():
    risk = RiskManager(Settings(_env_file=None, auto_execution=True, daily_loss_limit_percent=2))
    risk.daily_pnl = -1
    risk.initialize_live_equity(49)
    assert risk.daily_opening_balance == 50
    assert not risk.can_open_setup()
    risk.daily_pnl = -0.99
    assert risk.can_open_setup()


def test_live_day_rollover_captures_first_observed_equity():
    risk = RiskManager(Settings(_env_file=None, auto_execution=True))
    risk.daily_pnl = -2
    risk.initialize_live_equity(48)
    risk.day -= timedelta(days=1)
    assert risk.can_open_setup()
    assert risk.daily_pnl == 0
    assert risk.daily_opening_balance == 48
    assert risk.daily_opening_balance_source == "first_observed_exchange_equity_on_utc_day_proxy"
