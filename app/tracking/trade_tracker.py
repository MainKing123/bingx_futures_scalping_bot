from __future__ import annotations

import asyncio

from app.api.ws_manager import WSManager
from app.exchange.client import BingXClient
from app.schemas.setup import TradeSetup


class TradeTracker:
    def __init__(self, client: BingXClient, ws_manager: WSManager):
        self.client = client
        self.ws = ws_manager
        self.active: dict[str, TradeSetup] = {}

    async def add(self, setup: TradeSetup):
        self.active[setup.id] = setup

    async def track_loop(self):
        while True:
            await asyncio.sleep(2)
            for setup in list(self.active.values()):
                ticker = await self.client.get_ticker(setup.symbol)
                price = float(ticker.get("lastPrice", 0))
                if setup.direction == "LONG" and price <= setup.stop_loss:
                    setup.status = "SL_HIT"
                    await self.ws.broadcast("setup_update", setup.model_dump())
                    self.active.pop(setup.id, None)
                elif setup.direction == "SHORT" and price >= setup.stop_loss:
                    setup.status = "SL_HIT"
                    await self.ws.broadcast("setup_update", setup.model_dump())
                    self.active.pop(setup.id, None)
