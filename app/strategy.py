from __future__ import annotations

from app.models import Bias, MarketTick, PositionSide, StrategySignal


class ICTSMCStrategy:
    """Minimal ICT/SMC-inspired rules:
    - 30m bias by close location in range
    - 1m liquidity sweep + displacement trigger
    """

    def evaluate(self, tick: MarketTick) -> StrategySignal:
        range_30m = tick.high_30m - tick.low_30m
        if range_30m <= 0:
            return StrategySignal(has_signal=False, reason="Invalid 30m range")

        premium_threshold = tick.low_30m + range_30m * 0.65
        discount_threshold = tick.low_30m + range_30m * 0.35

        bias = Bias.neutral
        if tick.price >= premium_threshold:
            bias = Bias.bearish
        elif tick.price <= discount_threshold:
            bias = Bias.bullish

        if bias == Bias.neutral:
            return StrategySignal(has_signal=False, reason="No HTF bias", bias=bias)

        range_1m = tick.high_1m - tick.low_1m
        if range_1m <= 0:
            return StrategySignal(has_signal=False, reason="Invalid 1m range", bias=bias)

        displacement = range_1m / tick.price
        min_displacement = 0.0012  # 0.12%
        if displacement < min_displacement:
            return StrategySignal(
                has_signal=False,
                reason="No 1m displacement",
                bias=bias,
            )

        if bias == Bias.bullish:
            # emulate liquidity sweep below then reclaim
            entry = tick.price
            stop = min(tick.low_1m, tick.low_30m)
            tp = entry + (entry - stop) * 2
            return StrategySignal(
                has_signal=True,
                reason="Bullish sweep + displacement",
                bias=bias,
                side=PositionSide.long,
                entry_price=entry,
                stop_price=stop,
                take_profit=tp,
            )

        entry = tick.price
        stop = max(tick.high_1m, tick.high_30m)
        tp = entry - (stop - entry) * 2
        return StrategySignal(
            has_signal=True,
            reason="Bearish sweep + displacement",
            bias=bias,
            side=PositionSide.short,
            entry_price=entry,
            stop_price=stop,
            take_profit=tp,
        )
