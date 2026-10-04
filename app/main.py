from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.api.router import router
from app.config import Settings
from app.db.migrations import init_db
from app.exchange.client import MEXCClient
from app.execution.executor import AutoExecutor
from app.risk.risk_manager import RiskManager
from app.scanner.scanner import SignalScanner
from app.tracking.trade_tracker import TradeTracker
from app.universe import UniverseSelector


def create_app(settings=None, client=None, *, start_background=True):
    config = settings or Settings()
    if config.pair_selection == "fixed" and len(config.trading_symbols) != config.pair_count:
        raise ValueError("Fixed trading universe must contain exactly five distinct USDT pairs")
    @asynccontextmanager
    async def lifespan(app):
        if config.auto_execution and (not config.mexc_api_key or not config.mexc_api_secret):
            raise RuntimeError("Live mode needs local MEXC credentials")
        await init_db()
        exchange = client or MEXCClient(config)
        risk = RiskManager(config)
        executor = AutoExecutor(exchange, config)
        tracker = TradeTracker(exchange, config, risk, executor)
        universe = UniverseSelector(exchange, config) if config.pair_selection == "dynamic" else None
        scanner = SignalScanner(exchange, config, risk, tracker, executor, universe=universe)
        app.state.settings, app.state.client = config, exchange
        app.state.risk_manager, app.state.executor = risk, executor
        app.state.tracker, app.state.scanner = tracker, scanner
        try:
            await tracker.restore()
            if config.auto_execution:
                balance = await exchange.get_balance()
                risk.initialize_live_equity(balance["equity"])
                await tracker.poll()  # reconcile before enabling the scanner
                # Reconciliation may book more of today's PnL and refresh equity.
                # Its midnight estimate is explicitly a proxy, not the paper config.
                risk.initialize_live_equity(risk.balance)
            tasks = []
            if start_background:
                tasks = [asyncio.create_task(scanner.run()), asyncio.create_task(tracker.track_loop())]
            try:
                yield
            finally:
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with suppress(asyncio.CancelledError):
                        await task
        finally:
            if universe:
                await universe.close()
            await exchange.close()
    app = FastAPI(title="MEXC · VOLIUM", lifespan=lifespan)
    app.include_router(router)
    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    async def dashboard():
        return FileResponse(Path(__file__).parent / "web" / "static" / "index.html")
    return app


app = create_app()
