import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.router import router
from app.backtest.schemas import BacktestRunStatusResponse, BacktestSummary
from app.db.migrations import init_db
from app.schemas.analysis import CRTICTAnalysisResponse, CRTSnapshot, ICTSnapshot, LiquiditySnapshot, SessionSnapshot


def test_health_routes_without_lifespan_dependencies():
    asyncio.run(init_db())
    app = FastAPI()
    app.state.scanner = type("S", (), {"watchlist": {}, "volatile_pairs": []})()
    class DummyEngine:
        async def get_market_overview(self, symbol):
            return None

        async def analyze_crt_ict(self, symbol, ltf_timeframe=None):
            return (
                CRTICTAnalysisResponse(
                    symbol=symbol,
                    timeframe=ltf_timeframe or "5m",
                    timestamp=datetime.now(timezone.utc),
                    liquidity=LiquiditySnapshot(summary="ok"),
                    crt=CRTSnapshot(detected=False, summary="none"),
                    ict=ICTSnapshot(
                        htf_4h_bias="RANGING",
                        htf_1d_bias="RANGING",
                        htf_aligned=False,
                        mss_confirmed=False,
                        ob_confirmed=False,
                        fvg_confirmed=False,
                        summary="none",
                    ),
                    session=SessionSnapshot(killzone_enabled=True, killzone_active=False, summary="outside"),
                    signal="NO_SIGNAL",
                    targets=[],
                ),
                None,
            )

    app.state.engine = DummyEngine()
    app.state.client = type("C", (), {"get_balance": lambda self: {"balance": 0}, "get_positions": lambda self: []})()

    class DummyBacktestService:
        async def start(self, payload):
            return BacktestRunStatusResponse(
                job_id="job-1",
                status="queued",
                mode=payload.mode,
                strategy=payload.strategy,
                profile=payload.profile,
                progress=0.0,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                error=None,
            )

        def get_status(self, job_id: str):
            return BacktestRunStatusResponse(
                job_id=job_id,
                status="completed",
                mode="batch",
                strategy="crt_ict",
                profile="default",
                progress=1.0,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                error=None,
            )

        def get_result(self, job_id: str):
            return BacktestSummary(
                job_id=job_id,
                mode="batch",
                strategy="crt_ict",
                profile="default",
                lookback_days=14,
                ltf_timeframe="5m",
                htf_timeframe="30m",
                htf_context="1D+4H",
                universe=["BTC-USDT"],
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
                trades_count=1,
                wins=1,
                losses=0,
                win_rate=100,
                expectancy=1.2,
                profit_factor=2.5,
                max_drawdown=0.2,
                total_pnl_percent=1.2,
                symbol_results=[],
            )

        def get_latest_result(self):
            return self.get_result("job-1")

    app.state.backtest_service = DummyBacktestService()
    app.state.settings = type("Cfg", (), {
        "top_pairs_count": 20,
        "scan_interval_seconds": 300,
        "min_daily_volume_usd": 10_000_000,
        "volatility_pool_size": 60,
        "volatility_lookback_candles": 96,
        "volatility_interval": "5m",
        "max_poi_distance_pct": 0.5,
        "strategy_live_mode": "crt_shadow",
        "risk_per_trade_percent": 1.0,
        "max_open_setups": 5,
        "daily_loss_limit_percent": 3.0,
        "active_sessions": ["london"],
        "crt_killzone_enabled": True,
        "crt_london_session": (2, 5),
        "crt_new_york_session": (7, 10),
        "crt_range_lookback": 20,
        "crt_min_sweep_pct": 0.03,
        "crt_min_wick_body_ratio": 1.2,
        "crt_equal_level_tolerance": 0.0005,
        "crt_mss_lookback": 8,
        "crt_stop_buffer_bps": 2.0,
        "crt_min_rr": 2.0,
        "crt_entry_timeframes": ["5m", "15m"],
        "auto_execution": False,
        "ltf_timeframe": "1m",
        "backtest_fee_bps": 5.0,
        "backtest_slippage_bps": 2.0,
        "backtest_cooldown_candles": 3,
        "backtest_default_lookback_days": 14,
        "backtest_max_lookback_days": 30,
    })()
    app.include_router(router)
    client = TestClient(app)
    assert client.get('/api/stats').status_code == 200
    assert client.get('/api/watchlist').status_code == 200
    assert client.get('/api/scanner/volatile-pairs').status_code == 200
    assert client.get('/api/tradingview/symbol/BTC-USDT').status_code == 200
    assert client.get('/api/analysis/BTC-USDT').status_code == 200
    assert client.get('/api/analysis/BTC-USDT?ltf_timeframe=1m').status_code == 422
    assert client.post('/api/backtest/run', json={"mode": "batch"}).status_code == 200
    assert client.post('/api/backtest/run', json={"mode": "single"}).status_code == 422
    assert client.post('/api/backtest/run', json={"mode": "batch", "strategy": "crt_ict", "ltf_timeframe": "1m"}).status_code == 200
    assert client.get('/api/backtest/jobs/job-1').status_code == 200
    assert client.get('/api/backtest/jobs/job-1/result').status_code == 200
    assert client.get('/api/backtest/latest').status_code == 200
