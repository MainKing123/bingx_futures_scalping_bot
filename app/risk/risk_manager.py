from __future__ import annotations

from datetime import datetime, timezone

from app.config import Settings


class RiskManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.daily_pnl = 0.0
        self.open_setups = 0

    def calculate_position_size(self, entry: float, stop_loss: float, risk_percent: float, balance: float) -> float:
        risk_amount = balance * (risk_percent / 100)
        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return 0.0
        size = risk_amount / (stop_distance / entry)
        return max(0.0, round(size, 2))

    def can_open_setup(self) -> bool:
        if self.open_setups >= self.settings.max_open_setups:
            return False
        loss_limit = self.settings.account_balance_usdt * self.settings.daily_loss_limit_percent / 100
        return abs(min(0.0, self.daily_pnl)) < loss_limit

    def record_result(self, pnl: float) -> None:
        self.daily_pnl += pnl

    def get_daily_pnl(self) -> float:
        return round(self.daily_pnl, 4)

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0

    def utcnow(self) -> datetime:
        return datetime.now(timezone.utc)
