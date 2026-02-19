from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from loguru import logger

from app.api.ws_manager import WSManager
from app.config import Settings
from app.db.engine import SessionLocal
from app.db.repository import increment_daily_stats, update_setup_status
from app.exchange.client import BingXClient
from app.notifications.telegram import TelegramNotifier
from app.risk.risk_manager import RiskManager
from app.schemas.setup import TradeSetup


class TradeTracker:
    def __init__(self, client: BingXClient, ws_manager: WSManager, settings: Settings, risk_manager: RiskManager, notifier: TelegramNotifier):
        self.client = client
        self.ws = ws_manager
        self.settings = settings
        self.risk_manager = risk_manager
        self.notifier = notifier
        self.active: dict[str, TradeSetup] = {}
        self.added_at: dict[str, datetime] = {}

    async def add(self, setup: TradeSetup):
        self.active[setup.id] = setup
        self.added_at[setup.id] = datetime.now(timezone.utc)

    async def _close_setup(self, setup: TradeSetup, status: str, price: float):
        setup.status = status
        pnl_percent = ((price - setup.entry) / setup.entry * 100) if setup.direction == "LONG" else ((setup.entry - price) / setup.entry * 100)
        pnl_usdt = (setup.position_size_usdt or 0.0) * pnl_percent / 100
        self.risk_manager.record_result(pnl_usdt)
        self.risk_manager.open_setups = max(0, self.risk_manager.open_setups - 1)

        async with SessionLocal() as session:
            await update_setup_status(session, setup.id, status, pnl_percent)
            await increment_daily_stats(session, setup, pnl_percent)

        await self.ws.broadcast("setup_update", setup.model_dump())
        await self.notifier.send_status_update(setup, pnl_usdt)
        self.active.pop(setup.id, None)
        self.added_at.pop(setup.id, None)

    async def _process_prices(self, price_map: dict[str, float]):
        now = datetime.now(timezone.utc)
        for setup in list(self.active.values()):
            price = price_map.get(setup.symbol)
            if price is None:
                continue

            if now - self.added_at.get(setup.id, now) > timedelta(hours=self.settings.watchlist_max_age_hours):
                await self._close_setup(setup, "EXPIRED", price)
                continue

            if setup.direction == "LONG":
                if price <= setup.stop_loss:
                    await self._close_setup(setup, "SL_HIT", price)
                elif price >= setup.take_profits[2]:
                    await self._close_setup(setup, "TP3_HIT", price)
                elif price >= setup.take_profits[1]:
                    await self._close_setup(setup, "TP2_HIT", price)
                elif price >= setup.take_profits[0]:
                    await self._close_setup(setup, "TP1_HIT", price)
            else:
                if price >= setup.stop_loss:
                    await self._close_setup(setup, "SL_HIT", price)
                elif price <= setup.take_profits[2]:
                    await self._close_setup(setup, "TP3_HIT", price)
                elif price <= setup.take_profits[1]:
                    await self._close_setup(setup, "TP2_HIT", price)
                elif price <= setup.take_profits[0]:
                    await self._close_setup(setup, "TP1_HIT", price)

    async def track_loop(self):
        while True:
            await asyncio.sleep(2)
            if not self.active:
                continue
            try:
                tickers = await self.client.get_tickers()
                price_map = {t.get("symbol"): float(t.get("lastPrice") or 0) for t in tickers}
            except Exception as exc:
                logger.warning(f"Failed to load tickers in track loop: {exc}")
                continue
            await self._process_prices(price_map)
