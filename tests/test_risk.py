from datetime import datetime, timezone
import unittest

from app.models import RiskConfig
from app.risk import RiskManager


class DailyLossAccountingTests(unittest.TestCase):
    def test_daily_loss_accumulates_only_losses(self) -> None:
        risk = RiskConfig(equity_usdt=1000)
        manager = RiskManager()
        now = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

        used = manager.update_daily_loss_used_pct(100, risk, realized_pnl_usdt=100, now=now)
        self.assertEqual(used, 0.0)

        used = manager.update_daily_loss_used_pct(20, risk, realized_pnl_usdt=-80, now=now)
        self.assertEqual(used, 8.0)

    def test_daily_loss_resets_on_new_day(self) -> None:
        risk = RiskConfig(equity_usdt=1000)
        manager = RiskManager()
        day_one = datetime(2026, 1, 15, 23, 59, tzinfo=timezone.utc)
        day_two = datetime(2026, 1, 16, 0, 1, tzinfo=timezone.utc)

        used = manager.update_daily_loss_used_pct(-50, risk, realized_pnl_usdt=-50, now=day_one)
        self.assertEqual(used, 5.0)

        used = manager.update_daily_loss_used_pct(-50, risk, realized_pnl_usdt=0.0, now=day_two)
        self.assertEqual(used, 0.0)


if __name__ == "__main__":
    unittest.main()
