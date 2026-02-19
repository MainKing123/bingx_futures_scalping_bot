from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.router import router as api_router
from app.api.ws_manager import WSManager
from app.config import Settings
from app.db.migrations import init_db
from app.execution.executor import AutoExecutor
from app.exchange.client import BingXClient
from app.exchange.ws_stream import BingXKlineStream
from app.notifications.telegram import TelegramNotifier
from app.risk.risk_manager import RiskManager
from app.scanner.scanner import PairScanner
from app.strategy.smc_engine import SMCEngine
from app.tracking.trade_tracker import TradeTracker

settings = Settings()
ws_manager = WSManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    client = BingXClient(settings)
    kline_stream = BingXKlineStream()
    risk_manager = RiskManager(settings)
    notifier = TelegramNotifier(settings)
    tracker = TradeTracker(client, ws_manager, settings, risk_manager, notifier)
    engine = SMCEngine(client, settings, risk_manager)
    executor = AutoExecutor(client, settings)
    scanner = PairScanner(client, kline_stream, engine, settings, ws_manager, risk_manager, notifier, tracker, executor)

    app.state.settings = settings
    app.state.client = client
    app.state.stream = kline_stream
    app.state.engine = engine
    app.state.scanner = scanner
    app.state.risk_manager = risk_manager
    app.state.tracker = tracker
    app.state.notifier = notifier
    app.state.executor = executor

    scheduler = AsyncIOScheduler()
    scheduler.add_job(scanner.scan, "interval", seconds=settings.scan_interval_seconds)
    scheduler.add_job(risk_manager.reset_daily, "cron", hour=0, minute=0)
    scheduler.start()

    stream_task = asyncio.create_task(kline_stream.connect())
    scan_watch_task = asyncio.create_task(scanner.watch_loop())
    tracker_task = asyncio.create_task(tracker.track_loop())
    logger.info("Application startup complete")

    try:
        yield
    finally:
        scheduler.shutdown()
        for task in (stream_task, scan_watch_task, tracker_task):
            task.cancel()
        await kline_stream.close()
        await client.close()
        logger.info("Graceful shutdown complete")


app = FastAPI(title="SMC/ICT Crypto Scanner", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    finally:
        ws_manager.disconnect(websocket)
