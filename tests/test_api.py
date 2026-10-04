import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app import main
from app.config import Settings
from app.db.models import Base
from app.api import router as api
from app.execution import executor
from app.scanner import scanner
from app.tracking import trade_tracker


def test_local_dashboard_and_api_hide_credentials(monkeypatch):
    engine=create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions=async_sessionmaker(engine,expire_on_commit=False)
    async def init():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(main,"init_db",init)
    for module in (api,executor,scanner,trade_tracker):
        monkeypatch.setattr(module,"SessionLocal",sessions)
    settings=Settings(_env_file=None,mexc_api_key="test-key-only",mexc_api_secret="test-secret-only")
    client=SimpleNamespace(close=AsyncMock())
    app=main.create_app(settings,client,start_background=False)
    with TestClient(app) as browser:
        assert browser.get("/").status_code == 200
        config=browser.get("/api/config")
        assert config.status_code == 200
        assert "test-key-only" not in config.text
        assert "test-secret-only" not in config.text
        assert "mexc_api_key" not in config.json()
        status=browser.get("/api/status").json()
        assert status["mode"] == "paper"
        assert status["strategy"] == "intraday"
        assert browser.get("/api/setups").json() == []
        assert browser.get("/api/executions").json() == []
        assert browser.post("/api/setups/missing/cancel").status_code == 409
    client.close.assert_awaited_once()
    asyncio.run(engine.dispose())


def test_fixed_bot_requires_exactly_five_distinct_pairs():
    config = Settings(_env_file=None, pair_selection="fixed", trading_symbols=["BTC_USDT", "ETH_USDT"])
    with pytest.raises(ValueError, match="exactly five"):
        main.create_app(config, start_background=False)
