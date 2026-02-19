from __future__ import annotations

from aiogram import Bot

from app.config import Settings
from app.schemas.setup import TradeSetup


class TelegramNotifier:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bot = Bot(settings.telegram_bot_token) if settings.telegram_enabled and settings.telegram_bot_token else None

    async def send_setup(self, setup: TradeSetup):
        if not self.bot or not self.settings.telegram_chat_id:
            return
        text = (
            f"🟢 {setup.direction} | {setup.symbol}\n"
            f"Setup: {setup.setup_type} | Confidence: {setup.confidence}\n"
            f"Entry: {setup.entry:.4f} | SL: {setup.stop_loss:.4f}\n"
            f"TP: {', '.join(f'{x:.4f}' for x in setup.take_profits)}\n"
            f"RR: 1:{setup.risk_reward:.2f}"
        )
        await self.bot.send_message(self.settings.telegram_chat_id, text)
