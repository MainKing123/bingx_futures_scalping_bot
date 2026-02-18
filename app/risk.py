from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from app.models import Position, PositionSide, RiskConfig, StrategySignal


class RiskManager:
    def __init__(self) -> None:
        self._daily_loss_day: date | None = None
        self._daily_loss_usdt: float = 0.0

    def build_position(self, symbol: str, signal: StrategySignal, risk: RiskConfig) -> Position:
        if (
            not signal.has_signal
            or signal.entry_price is None
            or signal.stop_price is None
            or signal.take_profit is None
            or signal.side is None
        ):
            raise ValueError("Cannot build position from empty signal")

        if signal.side == PositionSide.long:
            buffered_stop = signal.stop_price * (1 - risk.stop_buffer_pct / 100)
        else:
            buffered_stop = signal.stop_price * (1 + risk.stop_buffer_pct / 100)

        stop_distance = abs(signal.entry_price - buffered_stop)
        if stop_distance <= 0:
            raise ValueError("Invalid stop distance")

        risk_usdt = risk.equity_usdt * (risk.risk_per_trade_pct / 100)
        position_notional = risk_usdt * (signal.entry_price / stop_distance)
        max_notional = risk.equity_usdt * risk.max_leverage
        size_usdt = min(position_notional, max_notional)

        leverage = max(1, min(risk.max_leverage, int(round(size_usdt / risk.equity_usdt))))
        margin_used = size_usdt / leverage

        if signal.side == PositionSide.long:
            tp1_price = signal.entry_price + (signal.entry_price - buffered_stop)
        else:
            tp1_price = signal.entry_price - (buffered_stop - signal.entry_price)

        return Position(
            symbol=symbol,
            side=PositionSide(signal.side),
            entry_price=signal.entry_price,
            stop_price=buffered_stop,
            initial_stop_price=buffered_stop,
            take_profit=signal.take_profit,
            tp1_price=tp1_price,
            size_usdt=round(size_usdt, 2),
            open_size_usdt=round(size_usdt, 2),
            leverage=leverage,
            margin_used=round(margin_used, 2),
        )

    def close_pnl(self, position: Position, close_price: float, size_usdt: float | None = None) -> float:
        notional = position.open_size_usdt if size_usdt is None else size_usdt
        if position.side == PositionSide.long:
            return round((close_price - position.entry_price) / position.entry_price * notional, 4)
        return round((position.entry_price - close_price) / position.entry_price * notional, 4)

    def apply_partial_take_profit(self, position: Position, tick_price: float) -> tuple[Position, float]:
        if position.open_size_usdt <= 0:
            return position, 0.0

        hit_tp1 = tick_price >= position.tp1_price if position.side == PositionSide.long else tick_price <= position.tp1_price
        if not hit_tp1 or position.moved_to_breakeven:
            return position, 0.0

        closed_size = round(position.open_size_usdt * 0.5, 2)
        pnl = self.close_pnl(position, tick_price, size_usdt=closed_size)
        position.open_size_usdt = round(position.open_size_usdt - closed_size, 2)
        position.stop_price = position.entry_price
        position.moved_to_breakeven = True
        position.trailing_active = True
        return position, pnl

    def apply_trailing_stop(self, position: Position, tick_price: float) -> Position:
        if not position.trailing_active:
            return position

        if position.side == PositionSide.long:
            candidate_stop = tick_price * 0.9985
            position.stop_price = max(position.stop_price, candidate_stop)
        else:
            candidate_stop = tick_price * 1.0015
            position.stop_price = min(position.stop_price, candidate_stop)
        return position

    def should_enter_cooldown(self, now: datetime, risk: RiskConfig) -> datetime:
        return now + timedelta(minutes=risk.cooldown_minutes)

    def update_daily_loss_used_pct(
        self,
        closed_pnl_usdt: float,
        risk: RiskConfig,
        realized_pnl_usdt: float | None = None,
        now: datetime | None = None,
    ) -> float:
        current_time = now or self.utcnow()
        current_day = current_time.date()

        if self._daily_loss_day != current_day:
            self._daily_loss_day = current_day
            self._daily_loss_usdt = 0.0

        if realized_pnl_usdt is None:
            # Backwards-compatible fallback for callers that only have a cumulative value.
            self._daily_loss_usdt = max(0.0, abs(min(0.0, closed_pnl_usdt)))
        elif realized_pnl_usdt < 0:
            self._daily_loss_usdt += abs(realized_pnl_usdt)

        return round(self._daily_loss_usdt / risk.equity_usdt * 100, 4)

    def utcnow(self) -> datetime:
        return datetime.now(timezone.utc)
