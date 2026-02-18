from __future__ import annotations

from app.models import Position, PositionSide, RiskConfig, StrategySignal


class RiskManager:
    def build_position(self, symbol: str, signal: StrategySignal, risk: RiskConfig) -> Position:
        if not signal.has_signal or signal.entry_price is None or signal.stop_price is None or signal.take_profit is None or signal.side is None:
            raise ValueError("Cannot build position from empty signal")

        stop_distance = abs(signal.entry_price - signal.stop_price)
        if stop_distance <= 0:
            raise ValueError("Invalid stop distance")

        risk_usdt = risk.equity_usdt * (risk.risk_per_trade_pct / 100)
        position_notional = risk_usdt * (signal.entry_price / stop_distance)

        max_notional = risk.equity_usdt * risk.max_leverage
        size_usdt = min(position_notional, max_notional)

        leverage = max(1, min(risk.max_leverage, int(round(size_usdt / risk.equity_usdt))))
        margin_used = size_usdt / leverage

        return Position(
            symbol=symbol,
            side=PositionSide(signal.side),
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            take_profit=signal.take_profit,
            size_usdt=round(size_usdt, 2),
            leverage=leverage,
            margin_used=round(margin_used, 2),
        )

    def close_pnl(self, position: Position, close_price: float) -> float:
        if position.side == PositionSide.long:
            return round((close_price - position.entry_price) / position.entry_price * position.size_usdt, 4)
        return round((position.entry_price - close_price) / position.entry_price * position.size_usdt, 4)
