import asyncio
from datetime import datetime, timezone

from app.api.ws_manager import WSManager
from app.config import Settings
from app.db.migrations import init_db
from app.notifications.telegram import TelegramNotifier
from app.risk.risk_manager import RiskManager
from app.schemas.setup import TradeSetup
from app.tracking.trade_tracker import TradeTracker


class DummyClient:
    async def get_tickers(self):
        return [{"symbol": "BTC-USDT", "lastPrice": "104"}]


def test_trade_tracker_hits_tp():
    async def run():
        await init_db()
        settings = Settings()
        tracker = TradeTracker(DummyClient(), WSManager(), settings, RiskManager(settings), TelegramNotifier(settings))
        setup = TradeSetup(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC-USDT",
            direction="LONG",
            setup_type="CHOCH_OB",
            htf_bias="BULLISH",
            entry=100,
            stop_loss=95,
            take_profits=[102, 103, 104],
            risk_reward=3,
            confidence="HIGH",
            confluences=["a", "b", "c", "d"],
            position_size_usdt=100,
        )
        await tracker.add(setup)
        await tracker._process_prices({"BTC-USDT": 104})
        assert setup.status == "TP3_HIT"
        assert setup.id not in tracker.active

    asyncio.run(run())
