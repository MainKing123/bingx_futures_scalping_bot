from __future__ import annotations

from aiogram import Bot

from app.config import Settings
from app.schemas.setup import TradeSetup


class TelegramNotifier:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bot = Bot(settings.telegram_bot_token) if settings.telegram_enabled and settings.telegram_bot_token else None

    async def _send(self, text: str):
        if not self.bot or not self.settings.telegram_chat_id:
            return
        await self.bot.send_message(self.settings.telegram_chat_id, text)

    async def send_setup(self, setup: TradeSetup):
        side_emoji = "🟢" if setup.direction == "LONG" else "🔴"
        sl_pct = abs((setup.entry - setup.stop_loss) / setup.entry * 100)
        confluences = "\n".join([f"• {c}" for c in setup.confluences])
        text = (
            f"━━━━━━━━━━━━━━━━━━\n"
            f"{side_emoji} {setup.direction} | {setup.symbol}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📊 Setup: {setup.setup_type} | Confidence: {setup.confidence}\n\n"
            f"▫️ Entry: {setup.entry:.4f}\n"
            f"🛑 Stop Loss: {setup.stop_loss:.4f} (-{sl_pct:.2f}%)\n"
            f"🎯 TP1: {setup.take_profits[0]:.4f}\n"
            f"🎯 TP2: {setup.take_profits[1]:.4f}\n"
            f"🎯 TP3: {setup.take_profits[2]:.4f}\n\n"
            f"💰 Risk/Reward: 1:{setup.risk_reward:.2f}\n"
            f"📐 Position Size: {setup.position_size_usdt or 0:.2f} USDT\n\n"
            f"✅ Confluences:\n{confluences}"
        )
        await self._send(text)

    async def send_status_update(self, setup: TradeSetup, pnl_usdt: float):
        emoji = "✅" if setup.status.startswith("TP") else "❌"
        await self._send(f"{emoji} {setup.symbol} {setup.status} | P&L: {pnl_usdt:.2f} USDT")

    async def send_daily_summary(self, stats: dict):
        await self._send(
            "📅 Daily summary\n"
            f"Setups: {stats.get('total_setups', 0)}\n"
            f"Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}\n"
            f"PnL: {stats.get('total_pnl_percent', 0):.2f}%"
        )
