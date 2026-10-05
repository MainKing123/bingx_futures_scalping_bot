"""Cost-aware fixed cash risk and isolated-margin leverage admission.

No price forecast or historical profitability selection occurs here. Contract
tiers are current public metadata. Liquidation distance is a conservative
entry-notional proxy, not an exchange-confirmed liquidation price.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import ceil, floor, isfinite

from pydantic import Field
from app.schemas.setup import TradeSetup
from app.exchange.client import contracts_for_notional, round_bracket_prices


class RuntimeTradeSetup(TradeSetup):
    leverage: int = Field(ge=10, le=50)
    economics: dict = {}
    strategy_version: str = "v1_cost_guard"


@dataclass(frozen=True)
class TradeEconomics:
    price_risk_fraction: float
    roundtrip_cost_fraction: float
    cost_to_price_risk: float
    gross_reward_risk: float
    net_reward_risk: float
    break_even_win_rate: float
    modeled_stop_loss_fraction: float


def _number(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError("Invalid " + name)
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError("Invalid " + name) from None
    if not isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError("Invalid " + name)
    return result


async def current_contract_with_basis(client, symbol):
    """Use observed mark/last difference, never a contract's variation coefficient."""
    contract = await client.get_contract(symbol, refresh=True)
    ticker = await client.get_ticker(symbol)
    if ticker.get("symbol") != symbol:
        raise ValueError("Fair-price observation belongs to another contract")
    last = _number(ticker.get("lastPrice"), "last price", positive=True)
    fair = _number(ticker.get("fairPrice"), "fair price", positive=True)
    stamp = _number(ticker.get("timestamp"), "fair-price observation time", positive=True) / 1000
    age = datetime.now(timezone.utc).timestamp() - stamp
    if not -5 <= age <= 90:
        raise ValueError("Fair-price observation is stale")
    return {**contract, "observedFairLastBasisFraction": abs(fair - last) / last,
            "fairLastBasisObservedAtMs": int(stamp * 1000)}


def trade_economics(setup, fee_bps, slippage_bps):
    entry = _number(setup.entry, "entry", positive=True)
    stop = _number(setup.stop_loss, "stop", positive=True)
    target = _number(setup.take_profits[0], "target", positive=True)
    if not (stop < entry < target if setup.direction == "LONG" else target < entry < stop):
        raise ValueError("Invalid bracket geometry")
    fee = _number(fee_bps, "fee") / 10000
    slip = _number(slippage_bps, "slippage") / 10000
    # Exit-side fees apply to exit notional. Reserve the worst bracket
    # turnover instead of understating costs when the exit price is higher.
    cost = (fee + slip) * (1 + max(entry, stop, target) / entry)
    risk, reward = abs(entry - stop) / entry, abs(target - entry) / entry
    net_win, net_loss = reward - cost, risk + cost
    return TradeEconomics(risk, cost, cost / risk, reward / risk,
                          net_win / net_loss, net_loss / (net_loss + net_win), net_loss)


def enforce_economics(setup, settings, *, fee_bps=None):
    result = trade_economics(setup, settings.paper_fee_bps if fee_bps is None else fee_bps,
                             settings.paper_slippage_bps)
    if result.price_risk_fraction * 100 > settings.max_stop_distance_percent + 1e-12:
        raise ValueError("Structural stop exceeds the configured distance; stop is not moved")
    if result.net_reward_risk + 1e-12 < settings.min_net_risk_reward:
        raise ValueError("Net reward/risk after modeled fees and slippage is too low")
    if result.cost_to_price_risk > settings.max_cost_to_price_risk + 1e-12:
        raise ValueError("Modeled trading costs consume too much of structural risk")
    return result


def _tier(contract, notional, entry):
    size = _number(contract.get("contractSize"), "contract size", positive=True)
    metric = notional if contract.get("riskLimitType", "BY_VOLUME") == "BY_VALUE" else notional / entry / size
    if contract.get("riskLimitType", "BY_VOLUME") not in {"BY_VOLUME", "BY_VALUE"}:
        raise ValueError("Unknown risk limit units")
    mode = contract.get("riskLimitMode", "INCREASE")
    if mode == "CUSTOM":
        tiers = contract.get("riskLimitCustom")
        if not isinstance(tiers, list) or not tiers:
            raise ValueError("Missing custom risk tiers")
        parsed = []
        for item in tiers:
            maximum = _number(item.get("maxVol"), "tier limit", positive=True)
            mmr = _number(item.get("mmr"), "maintenance rate")
            imr = _number(item.get("imr"), "initial margin rate", positive=True)
            leverage = _number(item.get("maxLeverage"), "tier leverage", positive=True)
            parsed.append((maximum, mmr, min(floor(leverage), floor(1 / imr))))
        for maximum, mmr, leverage in sorted(parsed):
            if metric <= maximum + 1e-9:
                return mmr, leverage
        raise ValueError("Position exceeds published custom risk tiers")
    if mode != "INCREASE":
        raise ValueError("Unknown risk tier mode")
    base = _number(contract.get("riskBaseVol"), "base risk limit", positive=True)
    increment = _number(contract.get("riskIncrVol"), "risk increment")
    level = 0 if metric <= base else ceil((metric - base) / increment) if increment else None
    count = _number(contract.get("riskLevelLimit"), "tier count", positive=True)
    if level is None or level >= count:
        raise ValueError("Position exceeds incremental risk tiers")
    mmr = _number(contract.get("maintenanceMarginRate"), "maintenance rate")
    imr = _number(contract.get("initialMarginRate"), "initial margin rate", positive=True)
    if level:
        mmr += level * _number(contract.get("riskIncrMmr"), "maintenance increment")
        imr += level * _number(contract.get("riskIncrImr"), "initial margin increment")
    return mmr, floor(1 / imr)


def choose_leverage(setup, settings, contract, notional, *, preferred_cap=None, fee_bps=None):
    """Try 50/25/20/10, then an intervening contract/user cap if necessary."""
    mmr, tier_cap = _tier(contract, notional, setup.entry)
    asset_cap = settings.leverage_asset_caps.get(setup.symbol, 10)
    ceiling = min(settings.default_leverage, asset_cap, tier_cap,
                  floor(_number(contract.get("maxLeverage"), "maximum leverage", positive=True)))
    country = _number(contract.get("countryConfigContractMaxLeverage", 0), "country maximum")
    if country:
        ceiling = min(ceiling, floor(country))
    if preferred_cap is not None:
        ceiling = min(ceiling, preferred_cap)
    minimum = max(10, ceil(_number(contract.get("minLeverage"), "minimum leverage", positive=True)))
    liquidation_fee = _number(contract.get("liquidationFeeRate"), "liquidation fee")
    # priceCoefficientVariation is a contract coefficient, not the currently
    # observed fair/last divergence. Historical replay has no observed value.
    fair_basis = max(settings.fair_price_basis_buffer_bps / 10000,
                     _number(contract.get("observedFairLastBasisFraction", 0), "observed fair/last basis"))
    economics = enforce_economics(setup, settings, fee_bps=fee_bps)
    reserve = economics.roundtrip_cost_fraction + settings.adverse_funding_reserve_bps / 10000
    stop_allowance = economics.price_risk_fraction + fair_basis + settings.liquidation_gap_buffer_bps / 10000
    candidates = sorted({ceiling, *[value for value in (50, 25, 20, 10) if value <= ceiling]}, reverse=True)
    for leverage in candidates:
        if leverage < minimum:
            continue
        distance = 1 / leverage - mmr - liquidation_fee - reserve
        if distance > 0 and stop_allowance < settings.liquidation_distance_safety_fraction * distance:
            return leverage, {
                "maintenance_margin_rate": mmr, "tier_leverage_cap": tier_cap,
                "conservative_liquidation_distance_fraction": distance,
                "stop_and_gap_allowance_fraction": stop_allowance,
                "fair_last_basis_buffer_fraction": fair_basis,
                "fair_last_basis_observed": "observedFairLastBasisFraction" in contract,
                "liquidation_price_is_proxy": True,
            }
    raise ValueError("No leverage from 10 to 50 leaves modeled room beyond the structural stop")


def prepare_runtime_setup(setup, settings, contract, equity, available_margin, *, preferred_cap=None):
    """Round first, size risk including modeled costs, then allocate margin."""
    if contract.get("apiAllowed") is not True or contract.get("state") != 0:
        raise ValueError("Contract does not currently allow API trading")
    if contract.get("positionOpenType") not in (1, 3):
        raise ValueError("Isolated margin unavailable")
    if contract.get("feeRateMode", "NORMAL") != "NORMAL":
        raise ValueError("Variable fee schedule needs explicit modeling before admission")
    entry, stop, target = round_bracket_prices(contract, setup.direction, setup.entry, setup.stop_loss, setup.take_profits[0])
    candidate = setup.model_copy(update={"entry": float(entry), "stop_loss": float(stop),
                                       "take_profits": [float(target)],
                                       "risk_reward": float(abs(target-entry) / abs(entry-stop))})
    fee_bps = max(settings.paper_fee_bps, 10000 * _number(contract.get("takerFeeRate"), "public taker fee"))
    economics = enforce_economics(candidate, settings, fee_bps=fee_bps)
    risk_budget = _number(equity, "equity", positive=True) * settings.risk_per_trade_percent / 100
    available = _number(available_margin, "available margin", positive=True)
    notional = risk_budget / economics.modeled_stop_loss_fraction
    maximum_volume = min(_number(contract.get("maxVol"), "max volume", positive=True),
                         _number(contract.get("limitMaxVol", contract.get("maxVol")), "limit max volume", positive=True))
    maximum_notional = maximum_volume * _number(contract.get("contractSize"), "contract size", positive=True) * candidate.entry
    notional = min(notional, maximum_notional)
    leverage, liquidity = choose_leverage(candidate, settings, contract, notional,
                                         preferred_cap=preferred_cap, fee_bps=fee_bps)
    # Include entry fee/slippage in the margin reservation. The 10% account
    # reserve is supplied by the caller, rather than applied twice here.
    notional = min(notional, available / (1 / leverage + (fee_bps + settings.paper_slippage_bps) / 10000))
    volume = contracts_for_notional(contract, notional, candidate.entry)
    notional = float(volume) * float(contract["contractSize"]) * candidate.entry
    values = asdict(economics)
    values.update(liquidity, modeled_fee_bps_per_side=fee_bps,
                  modeled_slippage_bps_per_side=settings.paper_slippage_bps,
                  risk_budget_usdt=risk_budget,
                  modeled_stop_loss_usdt=notional * economics.modeled_stop_loss_fraction,
                  initial_margin_usdt=notional / leverage,
                  reserved_margin_usdt=notional * (1 / leverage + (fee_bps + settings.paper_slippage_bps) / 10000),
                  modeled_target_profit_usdt=notional * (abs(candidate.take_profits[0]-candidate.entry) / candidate.entry
                                                       - economics.roundtrip_cost_fraction),
                  fees_are_current_public_proxy=True, funding_not_in_net_reward_risk=True)
    payload = candidate.model_dump()
    payload.update(leverage=leverage, economics=values, position_size_usdt=notional,
                   strategy_version=getattr(setup, "strategy_version", "v1_cost_guard"))
    return RuntimeTradeSetup.model_validate(payload)
