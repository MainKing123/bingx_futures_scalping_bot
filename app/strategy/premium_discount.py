from __future__ import annotations

from app.schemas.market import PremiumDiscount, SwingPoint


def get_premium_discount_zones(swings: list[SwingPoint]) -> PremiumDiscount:
    highs = [s.price for s in swings if s.type in {"HH", "LH"}] or [0.0]
    lows = [s.price for s in swings if s.type in {"HL", "LL"}] or [0.0]
    swing_high = max(highs)
    swing_low = min(lows)
    eq = (swing_high + swing_low) / 2
    return PremiumDiscount(
        swing_high=swing_high,
        swing_low=swing_low,
        equilibrium=eq,
        premium_top=swing_high,
        premium_bottom=eq,
        discount_top=eq,
        discount_bottom=swing_low,
    )


def is_in_discount(price: float, zones: PremiumDiscount) -> bool:
    return zones.discount_bottom <= price <= zones.discount_top


def is_in_premium(price: float, zones: PremiumDiscount) -> bool:
    return zones.premium_bottom <= price <= zones.premium_top
