import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.router import router
from app.db.migrations import init_db


def test_health_routes_without_lifespan_dependencies():
    asyncio.run(init_db())
    app = FastAPI()
    app.state.scanner = type("S", (), {"watchlist": {}})()
    app.state.engine = type("E", (), {"get_market_overview": lambda self, symbol: None})()
    app.state.client = type("C", (), {"get_balance": lambda self: {"balance": 0}, "get_positions": lambda self: []})()
    app.state.settings = type("Cfg", (), {
        "top_pairs_count": 30,
        "scan_interval_seconds": 300,
        "min_daily_volume_usd": 10_000_000,
        "risk_per_trade_percent": 1.0,
        "max_open_setups": 5,
        "daily_loss_limit_percent": 3.0,
        "active_sessions": ["london"],
        "auto_execution": False,
    })()
    app.include_router(router)
    client = TestClient(app)
    assert client.get('/api/stats').status_code == 200
    assert client.get('/api/watchlist').status_code == 200
