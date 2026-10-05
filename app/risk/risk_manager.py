import math
from datetime import datetime, timezone
from app.config import Settings


class RiskManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.daily_pnl = 0.0
        self.open_setups = 0
        self.day = datetime.now(timezone.utc).date()
        self.halted = False
        self.balance = settings.account_balance_usdt
        self.daily_opening_balance = self.balance
        self.daily_opening_balance_source = "configured_paper_balance"

    def initialize_live_equity(self, equity):
        """Estimate today's opening equity from current equity and the bot's booked PnL.

        This is a proxy: deposits, withdrawals, other trading and unrealized PnL
        prevent recovery of the account's exact midnight equity from these inputs.
        """
        equity = float(equity)
        if not math.isfinite(equity) or not math.isfinite(self.daily_pnl):
            raise ValueError("Live equity and restored daily PnL must be finite")
        self._new_day()
        self.balance = equity
        self.daily_opening_balance = max(equity - self.daily_pnl, 0.0)
        self.daily_opening_balance_source = "exchange_equity_minus_today_bot_pnl_proxy"

    def calculate_position_size(self, entry, stop_loss, risk_percent, balance):
        if entry <= 0 or abs(entry-stop_loss) <= 0:
            return 0.0
        return max(0.0, balance * risk_percent / 100 * entry / abs(entry-stop_loss))

    def can_open_setup(self):
        self._new_day()
        return (math.isfinite(self.balance) and self.balance > 0
            and math.isfinite(self.daily_opening_balance) and self.daily_opening_balance > 0
            and math.isfinite(self.daily_pnl)
            and not self.halted and self.open_setups < self.settings.max_open_setups
            and self.daily_pnl > -self.daily_opening_balance * self.settings.daily_loss_limit_percent / 100)

    def _new_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.daily_pnl = 0.0
            self.day = today
            self.daily_opening_balance = self.balance
            self.daily_opening_balance_source = (
                "first_observed_exchange_equity_on_utc_day_proxy"
                if self.settings.auto_execution else "paper_balance_on_utc_day"
            )

    def record_result(self, pnl, *, closed_at=None):
        self._new_day()
        if closed_at is None:
            result_day = self.day
        else:
            if closed_at.tzinfo is None:
                closed_at = closed_at.replace(tzinfo=timezone.utc)
            result_day = closed_at.astimezone(timezone.utc).date()
        if result_day == self.day:
            self.daily_pnl += pnl
        elif result_day < self.day and not self.settings.auto_execution:
            # Recovered historical paper results belonged to the balance already
            # available at today's opening, even though they were booked late.
            self.daily_opening_balance += pnl
            self.daily_opening_balance_source = "restored_paper_ledger"

    def get_daily_pnl(self):
        self._new_day()
        return round(self.daily_pnl, 4)
