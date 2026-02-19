from __future__ import annotations

from loguru import logger

from app.config import Settings
from app.db.engine import SessionLocal
from app.db.models import SetupRecord
from app.exchange.client import BingXClient
from app.schemas.setup import TradeSetup


class AutoExecutor:
    def __init__(self, client: BingXClient, settings: Settings):
        self.client = client
        self.settings = settings

    async def execute_setup(self, setup: TradeSetup) -> None:
        if not self.settings.auto_execution:
            return
        side = "BUY" if setup.direction == "LONG" else "SELL"
        position_side = "LONG" if setup.direction == "LONG" else "SHORT"
        close_side = "SELL" if setup.direction == "LONG" else "BUY"
        try:
            await self.client.set_leverage(setup.symbol, position_side, self.settings.default_leverage)
            entry = await self.client.place_order(setup.symbol, side, position_side, "MARKET", setup.position_size_usdt or 0)
            sl = await self.client.place_order(setup.symbol, close_side, position_side, "STOP_MARKET", setup.position_size_usdt or 0, stop_price=setup.stop_loss)
            tp = await self.client.place_order(setup.symbol, close_side, position_side, "TAKE_PROFIT_MARKET", setup.position_size_usdt or 0, stop_price=setup.take_profits[0])
            await self._save_order_ids(setup.id, str(entry.get("orderId")), str(sl.get("orderId")), str(tp.get("orderId")))
        except Exception as exc:
            logger.exception(f"Auto execution failed for {setup.id}: {exc}")

    async def cancel_exit_orders(self, setup: SetupRecord):
        if not self.settings.auto_execution:
            return
        if setup.sl_order_id:
            await self.client.cancel_order(setup.symbol, setup.sl_order_id)
        if setup.tp_order_id:
            await self.client.cancel_order(setup.symbol, setup.tp_order_id)

    async def _save_order_ids(self, setup_id: str, entry_id: str, sl_id: str, tp_id: str):
        async with SessionLocal() as session:
            setup = await session.get(SetupRecord, setup_id)
            if setup is None:
                return
            setup.entry_order_id = entry_id
            setup.sl_order_id = sl_id
            setup.tp_order_id = tp_id
            await session.commit()
