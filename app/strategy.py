from __future__ import annotations

from app.models import Bias, MarketTick, PositionSide, StrategyConfig, StrategySignal


class ICTSMCStrategy:
    """Simple ICT/SMC-inspired model.

    HTF (30m): infer bias from premium/discount placement.
    LTF (1m): require displacement and construct entry/SL/TP.
    """

    def evaluate(self, tick: MarketTick, config: StrategyConfig) -> StrategySignal:
        range_30m = tick.high_30m - tick.low_30m
        if range_30m <= 0:
            return StrategySignal(has_signal=False, reason="Invalid 30m range")

        premium_threshold = tick.low_30m + range_30m * config.premium_zone
        discount_threshold = tick.low_30m + range_30m * config.discount_zone

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

        displacement_pct = (range_1m / tick.price) * 100
        if displacement_pct < config.min_displacement_pct:
            return StrategySignal(has_signal=False, reason="No 1m displacement", bias=bias)

        if bias == Bias.bullish:
            entry = tick.price
            stop = min(tick.low_1m, tick.low_30m)
            risk = entry - stop
            if risk <= 0:
                return StrategySignal(has_signal=False, reason="Invalid long stop", bias=bias)
            tp = entry + risk * config.min_rr
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
        risk = stop - entry
        if risk <= 0:
            return StrategySignal(has_signal=False, reason="Invalid short stop", bias=bias)
        tp = entry - risk * config.min_rr
        return StrategySignal(
            has_signal=True,
            reason="Bearish sweep + displacement",
            bias=bias,
            side=PositionSide.short,
            entry_price=entry,
            stop_price=stop,
            take_profit=tp,
        )
