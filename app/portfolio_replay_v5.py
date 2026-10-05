"""Versioned V5 cash-cost replay; historical signed engines remain unchanged."""
from __future__ import annotations

from dataclasses import asdict
from math import floor, isfinite

import numpy as np
import pandas as pd

from app.portfolio_replay import _frame, _utc
from app.replay import SECONDS, required_timeframes
from app.strategy.volium import analyze_volium_from_df, in_volium_session
from app.execution.economics import RuntimeTradeSetup, enforce_economics, prepare_runtime_setup


def reservation(order):
    """Fees are reserved until paid, never reserved a second time after fill."""
    result = order["notional"] / order["leverage"]
    return result if order["filled"] else result + order["notional"]*(order["fee_rate"]+order["slip_rate"])


def mexc_admission(contracts):
    def admit(setup, settings, *, equity, available_margin):
        return prepare_runtime_setup(setup, settings, contracts[setup.symbol], equity, available_margin)
    admit.policy = {"venue":"mexc", "contract_rules":"current public snapshot, historical proxy"}
    return admit


def binance_proxy_admission(contracts, *, maintenance_margin_rate=.01):
    """Explicit research-only Binance proxy: no MEXC tick, lot or risk tier.

    Public liquidation fee is real current metadata. Signed historical leverage
    brackets are unavailable; 1% maintenance is a predeclared conservative proxy.
    Execution notional stays continuous and uses the same account-risk guard.
    """
    if not isfinite(maintenance_margin_rate) or not 0 <= maintenance_margin_rate < .1:
        raise ValueError("Invalid maintenance margin proxy")
    def admit(setup, settings, *, equity, available_margin):
        economics = enforce_economics(setup, settings)
        liquidation_fee = float(contracts[setup.symbol]["liquidationFee"])
        if not isfinite(liquidation_fee) or not 0 <= liquidation_fee < .1:
            raise ValueError("Invalid public Binance liquidation fee")
        ceiling = min(settings.default_leverage, settings.leverage_asset_caps.get(setup.symbol,10),50)
        costs = economics.roundtrip_cost_fraction + settings.adverse_funding_reserve_bps/10000
        fair_basis = getattr(settings, "fair_price_basis_buffer_bps",20)/10000
        allowance = economics.price_risk_fraction + fair_basis + settings.liquidation_gap_buffer_bps/10000
        candidates = sorted({floor(ceiling), *[value for value in (50,25,20,10) if value <= ceiling]}, reverse=True)
        leverage = next((value for value in candidates if value >=10 and allowance < settings.liquidation_distance_safety_fraction*(1/value-maintenance_margin_rate-liquidation_fee-costs)),None)
        if leverage is None:
            raise ValueError("No research proxy leverage leaves room beyond the structural stop")
        risk_budget = equity * settings.risk_per_trade_percent/100
        entry_cost = (settings.paper_fee_bps+settings.paper_slippage_bps)/10000
        notional = min(risk_budget/economics.modeled_stop_loss_fraction,available_margin/(1/leverage+entry_cost))
        if not isfinite(notional) or notional <=0:
            raise ValueError("No available margin")
        values = asdict(economics)
        values.update(modeled_fee_bps_per_side=settings.paper_fee_bps,
            modeled_slippage_bps_per_side=settings.paper_slippage_bps,risk_budget_usdt=risk_budget,
            modeled_stop_loss_usdt=notional*economics.modeled_stop_loss_fraction,
            initial_margin_usdt=notional/leverage,reserved_margin_usdt=notional*(1/leverage+entry_cost),
            maintenance_margin_rate=maintenance_margin_rate,public_liquidation_fee=liquidation_fee,
            leverage_brackets_are_proxy=True,liquidation_price_is_proxy=True,
            stop_and_gap_allowance_fraction=allowance,
            conservative_liquidation_distance_fraction=1/leverage-maintenance_margin_rate-liquidation_fee-costs,
            historical_tick_lot_tiers_not_applied=True)
        payload = setup.model_dump()
        payload.update(leverage=leverage,economics=values,position_size_usdt=notional,strategy_version="v5_research")
        return RuntimeTradeSetup.model_validate(payload)
    admit.policy = {"venue":"binance_usdm", "maintenance_margin_rate_proxy":maintenance_margin_rate,
        "public_liquidation_fee_applied":True,"historical_tick_lot_tiers_not_applied":True}
    return admit


def portfolio_replay_v5(frames_by_symbol, settings, *, symbols=None, start_at=None, end_at=None,
                     execution_frames=None, funding_rates=None, signal_provider=analyze_volium_from_df,
                     execution_timeframe="1m", admission=None, signal_schedule=None, history_bars=None):
    """Explicit cost-aware shared wallet; never mutates the frozen replay.

    Admission is a venue-specific pure callable returning RuntimeTradeSetup.
    Entry/exit fee and cash slippage use their respective actual notionals.
    Pending orders reserve margin plus entry cost; filled positions reserve
    initial margin. New orders cannot borrow unrealized inventory losses.
    Funding, OHLC causality, ambiguity and expiry follow the frozen engine.
    Liquidation distance is an admission proxy, not historical mark-price
    liquidation reconstruction. Price gaps can exceed the planned cash risk.
    """
    if admission is None:
        raise ValueError("V5 requires an explicit venue-specific admission adapter")
    history_bars = history_bars or {}
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in history_bars.values()):
        raise ValueError("History bars must be positive integers")
    if settings.volium_mode not in {"intraday", "scalp"}:
        raise ValueError("Portfolio replay supports intraday/scalp modes")
    if execution_timeframe not in {"1m", "5m"}:
        raise ValueError("Portfolio execution timeframe must be 1m or 5m")
    duration = pd.Timedelta(seconds=SECONDS[execution_timeframe])
    frequency = f"{SECONDS[execution_timeframe]}s"
    label = "M1" if execution_timeframe == "1m" else "M5"
    symbols = list(symbols or frames_by_symbol)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Portfolio symbols must be unique and ordered")
    if signal_schedule is not None and set(signal_schedule) != set(symbols):
        raise ValueError("Positive signal schedule must explicitly cover every portfolio symbol")
    timeframes = required_timeframes(settings)
    if SECONDS[execution_timeframe] > SECONDS[timeframes[-1]]:
        raise ValueError("Execution cannot be coarser than the strategy confirmation timeframe")
    strategy, execution, values, close_indices, signal_times, rates = {}, {}, {}, {}, {}, {}
    rate_cursor = {symbol: 0 for symbol in symbols}
    clocks = pd.DatetimeIndex([], tz="UTC")
    for symbol in symbols:
        if symbol not in frames_by_symbol or not set(timeframes).issubset(frames_by_symbol[symbol]):
            raise ValueError(f"Missing strategy frames for {symbol}")
        strategy[symbol] = {tf: _frame(frames_by_symbol[symbol][tf]) for tf in timeframes}
        source = execution_frames[symbol] if execution_frames is not None else frames_by_symbol[symbol].get(execution_timeframe)
        if source is None:
            raise ValueError(f"Missing {label} execution frame for {symbol}")
        execution[symbol] = _frame(source)
        values[symbol] = execution[symbol][["open", "high", "low", "close"]].to_numpy(dtype=float)
        if not execution[symbol].index.equals(execution[symbol].index.floor(frequency)):
            raise ValueError(f"{label} execution timestamps must align to {execution_timeframe} boundaries")
        clocks = clocks.union(execution[symbol].index)
        close_indices[symbol] = {tf: frame.index + pd.Timedelta(seconds=SECONDS[tf])
                                 for tf, frame in strategy[symbol].items()}
        signal_times[symbol] = set(close_indices[symbol][timeframes[-1]].asi8)
        if signal_schedule is not None:
            declared = [_utc(value).value for value in signal_schedule[symbol]]
            if len(set(declared)) != len(declared) or not set(declared).issubset(signal_times[symbol]):
                raise ValueError("Signal schedule must contain unique observed confirmation closes")
            if any((start_at is not None and value <= _utc(start_at).value)
                   or (end_at is not None and value > _utc(end_at).value) for value in declared):
                raise ValueError("Positive signal schedule extends beyond the research window")
            signal_times[symbol] = set(declared)
        series = None if funding_rates is None else funding_rates.get(symbol)
        rates[symbol] = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC")) if series is None else series.copy()
        rates[symbol].index = pd.to_datetime(rates[symbol].index, utc=True)
        rates[symbol] = rates[symbol].loc[~rates[symbol].index.duplicated(keep="last")].sort_index().astype(float)
        if not np.isfinite(rates[symbol].to_numpy()).all():
            raise ValueError("Funding rates must be finite")
    if start_at is not None:
        clocks = clocks[clocks >= _utc(start_at)]
    if end_at is not None:
        clocks = clocks[clocks + duration <= _utc(end_at)]
    if clocks.empty:
        raise ValueError("No execution bars in the requested window")
    first = _utc(start_at).ceil(frequency) if start_at is not None else clocks[0]
    last = _utc(end_at).floor(frequency)-duration if end_at is not None else clocks[-1]
    clocks = pd.date_range(first,last,freq=frequency)
    if clocks.empty:
        raise ValueError("No complete execution bars in the requested window")
    positions = {symbol: execution[symbol].index.get_indexer(clocks) for symbol in symbols}
    for symbol in symbols:
        missing = int((positions[symbol]<0).sum())
        if missing:
            raise ValueError(f"Missing {missing} {label} execution bars for {symbol}; portfolio replay aborted")
    execution_closes = {symbol: frame.index + duration for symbol, frame in execution.items()}
    initial = float(settings.account_balance_usdt)
    equity, marked_equity, realized_peak, marked_peak = initial, initial, initial, initial
    max_realized_dd = max_marked_dd = unrealized = 0.0
    active, accepted_ids, observed_ids, trades, last_marks, daily_close = {}, set(), set(), [], {}, {}
    counts = {key: 0 for key in ("signals", "accepted_signals", "rejected_slots", "rejected_margin",
              "rejected_geometry", "expired_limits", "ambiguous_expiry_limits", "daily_blocked_signal_checks",
              "funding_events_charged", "funding_events_skipped_favorable", "downsized_for_margin")}
    counts.update({"rejected_admission": 0, "gap_losses_beyond_planned_risk": 0,
                   "entry_cost_events": 0, "exit_cost_events": 0})
    admission_rejections = {}
    fees_total = slippage_total = gross_price_total = 0.0
    symbol_cash = {symbol: 0.0 for symbol in symbols}
    funding_total = 0.0
    symbol_funding = {symbol: 0.0 for symbol in symbols}
    symbol_signals = {symbol: 0 for symbol in symbols}
    symbol_accepted = {symbol: 0 for symbol in symbols}
    peak_active = 0
    peak_margin = peak_margin_utilization = 0.0
    daily_pnl, day, daily_opening_equity = 0.0, None, initial
    for step, opened in enumerate(clocks):
        closed = opened + duration
        if closed.date() != day:
            day, daily_pnl, daily_opening_equity = closed.date(), 0.0, equity
        rows = {}
        for symbol in symbols:
            i = positions[symbol][step]
            if i < 0:
                continue
            open_price, high, low, close_price = values[symbol][i]
            rows[symbol] = (open_price, high, low, close_price)
            last_marks[symbol] = float(close_price)
            events = []
            series = rates[symbol]
            while rate_cursor[symbol] < len(series) and series.index[rate_cursor[symbol]] <= closed:
                n = rate_cursor[symbol]
                events.append((series.index[n], float(series.iloc[n])))
                rate_cursor[symbol] += 1
            order = active.get(symbol)
            if order is None:
                continue
            setup = order["setup"]
            long = setup.direction == "LONG"
            opened_through = open_price <= setup.entry if long else open_price >= setup.entry
            filled_this_bar = False
            if not order["filled"]:
                expiry = _utc(setup.timestamp) + pd.Timedelta(minutes=settings.pending_order_max_age_minutes)
                touched = low <= setup.entry <= high
                if opened >= expiry:
                    del active[symbol]
                    counts["expired_limits"] += 1
                    continue
                if opened_through or (touched and closed <= expiry):
                    order.update(filled=True, entry_at=opened, entry_time=opened.isoformat())
                    filled_this_bar = True
                    entry_fee = order["notional"] * order["fee_rate"]
                    entry_slip = order["notional"] * order["slip_rate"]
                    order.update(entry_fee=entry_fee, entry_slippage=entry_slip)
                    equity -= entry_fee + entry_slip
                    daily_pnl -= entry_fee + entry_slip
                    symbol_cash[symbol] -= entry_fee + entry_slip
                    fees_total += entry_fee
                    slippage_total += entry_slip
                    counts["entry_cost_events"] += 1
                elif closed > expiry:
                    counts["ambiguous_expiry_limits"] += int(touched)
                    counts["expired_limits"] += 1
                    del active[symbol]
                    continue
            if not order["filled"]:
                continue
            stop_hit = low <= setup.stop_loss if long else high >= setup.stop_loss
            target_hit = high >= setup.take_profits[0] if long else low <= setup.take_profits[0]
            if filled_this_bar and not opened_through:
                target_hit = False
            sign = 1 if long else -1
            exit_at_open = ((stop_hit and (open_price <= setup.stop_loss if long else open_price >= setup.stop_loss))
                            or (target_hit and (open_price >= setup.take_profits[0] if long else open_price <= setup.take_profits[0])))
            for settlement, rate in events:
                if settlement < order["entry_at"] or (exit_at_open and settlement > opened):
                    continue
                mark_i = execution_closes[symbol].searchsorted(settlement, side="right") - 1
                mark = float(values[symbol][mark_i, 3]) if mark_i >= 0 else float(open_price)
                cash = -sign * rate * order["notional"] / setup.entry * mark
                known_held = not (stop_hit or target_hit) and (not filled_this_bar or (opened_through and settlement > opened))
                if cash > 0 and not known_held:
                    counts["funding_events_skipped_favorable"] += 1
                    continue
                if cash:
                    order["funding"] += cash
                    funding_total += cash
                    symbol_funding[symbol] += cash
                    symbol_cash[symbol] += cash
                    equity += cash
                    daily_pnl += cash
                    counts["funding_events_charged"] += 1
            if stop_hit or target_hit:
                exit_price = setup.stop_loss if stop_hit else setup.take_profits[0]
                if stop_hit:
                    exit_price = min(exit_price, open_price) if long else max(exit_price, open_price)
                gross = order["notional"] * sign * (exit_price - setup.entry) / setup.entry
                exit_notional = order["notional"] / setup.entry * exit_price
                exit_fee = exit_notional * order["fee_rate"]
                exit_slip = exit_notional * order["slip_rate"]
                pnl_cash = gross - exit_fee - exit_slip
                net_price = pnl_cash - order["entry_fee"] - order["entry_slippage"]
                equity += pnl_cash
                daily_pnl += pnl_cash
                symbol_cash[symbol] += pnl_cash
                gross_price_total += gross
                fees_total += exit_fee
                slippage_total += exit_slip
                counts["exit_cost_events"] += 1
                counts["gap_losses_beyond_planned_risk"] += int(-net_price > order["risk_budget"] + 1e-8)
                total_pnl = net_price + order["funding"]
                margin = order["notional"] / order["leverage"]
                trades.append({"symbol": symbol, "id": setup.id, "signal_time":setup.timestamp.isoformat(),
                    "entry_time":order["entry_time"], "exit_time":closed.isoformat(), "direction":setup.direction,
                    "entry":setup.entry, "stop":setup.stop_loss, "target":setup.take_profits[0], "exit":float(exit_price),
                    "notional":order["notional"], "leverage":order["leverage"],
                    "margin_reserved_usdt":margin, "gross_price_pnl_usdt":gross,
                    "entry_fee_usdt":order["entry_fee"], "exit_fee_usdt":exit_fee,
                    "entry_slippage_usdt":order["entry_slippage"], "exit_slippage_usdt":exit_slip,
                    "price_pnl_usdt":net_price, "funding_usdt":order["funding"], "pnl_usdt":total_pnl,
                    "margin_roi_percent":total_pnl/margin*100, "cash_risk_budget_usdt":order["risk_budget"],
                    "economics":setup.economics, "status":"SL_HIT" if stop_hit else "TP1_HIT"})
                del active[symbol]
        # All pairs' minute exits and funding have settled before sizing any order.
        realized_peak = max(realized_peak, equity)
        max_realized_dd = max(max_realized_dd, (realized_peak - equity) / realized_peak * 100)
        unrealized = sum((1 if order["setup"].direction == "LONG" else -1) * order["notional"]
            * (last_marks[symbol] - order["setup"].entry) / order["setup"].entry
            for symbol, order in active.items() if order["filled"])
        marked_equity = equity + unrealized
        marked_peak = max(marked_peak, marked_equity)
        max_marked_dd = max(max_marked_dd, (marked_peak - marked_equity) / marked_peak * 100)
        for symbol in symbols:
            if symbol not in rows or symbol in active or closed.value not in signal_times[symbol]:
                continue
            if settings.volium_session_enabled and not in_volium_session(closed, settings):
                continue
            if equity <= 0 or daily_pnl <= -daily_opening_equity * settings.daily_loss_limit_percent / 100:
                counts["daily_blocked_signal_checks"] += 1
                continue
            known = {tf: frame.iloc[:close_indices[symbol][tf].searchsorted(closed, side="right")].tail(history_bars.get(tf, settings.volium_context_lookback+30))
                     for tf, frame in strategy[symbol].items()}
            setup = signal_provider(symbol=symbol, frames=known, settings=settings, mode=settings.volium_mode, now=closed)
            if setup is None or (symbol, setup.id) in accepted_ids:
                continue
            if setup.symbol != symbol or _utc(setup.timestamp) != closed:
                raise ValueError("Signal must belong to the symbol and current confirmed candle close")
            if (symbol,setup.id) not in observed_ids:
                observed_ids.add((symbol,setup.id))
                counts["signals"] += 1
                symbol_signals[symbol] += 1
            if len(active) >= settings.max_open_setups:
                counts["rejected_slots"] += 1
                continue
            reserved = sum(reservation(order) for order in active.values())
            available = max(0.0, min(equity, marked_equity) * .9 - reserved)
            if available <= 1e-9:
                counts["rejected_margin"] += 1
                continue
            try:
                prepared = admission(setup, settings, equity=equity, available_margin=available)
            except ValueError as exc:
                counts["rejected_admission"] += 1
                reason = str(exc)
                admission_rejections[reason] = admission_rejections.get(reason, 0) + 1
                continue
            if prepared.symbol != symbol or _utc(prepared.timestamp) != closed or prepared.id != setup.id or prepared.direction != setup.direction:
                raise ValueError("Admission must preserve signal identity and timestamp")
            notional = float(prepared.position_size_usdt)
            leverage = float(prepared.leverage)
            fee_rate = float(prepared.economics["modeled_fee_bps_per_side"]) / 10000
            slip_rate = float(prepared.economics["modeled_slippage_bps_per_side"]) / 10000
            if not all(np.isfinite(value) for value in (notional, leverage, fee_rate, slip_rate)) or notional <= 0 or not 10 <= leverage <= 50 or min(fee_rate, slip_rate) < 0:
                raise ValueError("Invalid admission size, leverage or costs")
            if fee_rate*10000 + 1e-10 < settings.paper_fee_bps or abs(slip_rate*10000-settings.paper_slippage_bps) > 1e-10:
                raise ValueError("Admission costs must retain the preregistered fee floor and slippage")
            needed = notional * (1/leverage + fee_rate + slip_rate)
            if needed > available + 1e-7:
                raise ValueError("Admission borrowed unavailable shared margin")
            risk_budget = equity * settings.risk_per_trade_percent / 100
            economics = enforce_economics(prepared, settings, fee_bps=fee_rate*10000)
            if notional * economics.modeled_stop_loss_fraction > risk_budget + 1e-7:
                raise ValueError("Admission exceeds fee-inclusive cash risk")
            wanted = risk_budget / economics.modeled_stop_loss_fraction
            counts["downsized_for_margin"] += int(notional + 1e-7 < wanted and available < wanted*(1/leverage+fee_rate+slip_rate))
            active[symbol] = {"setup":prepared, "notional":notional, "leverage":leverage,
                "fee_rate":fee_rate, "slip_rate":slip_rate, "risk_budget":risk_budget,
                "filled":False, "funding":0.0, "entry_fee":0.0, "entry_slippage":0.0}
            accepted_ids.add((symbol,setup.id))
            counts["accepted_signals"] += 1
            symbol_accepted[symbol] += 1
        reserved = sum(reservation(order) for order in active.values())
        peak_active = max(peak_active, len(active))
        peak_margin = max(peak_margin, reserved)
        if equity > 0:
            peak_margin_utilization = max(peak_margin_utilization, reserved / equity * 100)
        daily_close[str(day)] = {"date_utc":str(day), "opening_equity":daily_opening_equity,
            "realized_equity":equity, "marked_equity":marked_equity, "daily_realized_pnl":daily_pnl,
            "active_orders_positions":len(active), "reserved_margin_usdt":reserved}
    profits = sum(max(trade["pnl_usdt"], 0) for trade in trades)
    losses = -sum(min(trade["pnl_usdt"], 0) for trade in trades)
    wins = sum(trade["pnl_usdt"] > 0 for trade in trades)
    unfinished_positions = sum(order["filled"] for order in active.values())
    symbol_results = []
    for symbol in symbols:
        own = [trade for trade in trades if trade["symbol"] == symbol]
        symbol_results.append({"symbol":symbol, "signals":symbol_signals[symbol],
            "accepted_signals":symbol_accepted[symbol], "trades_count":len(own),
            "price_pnl_usdt":sum(trade["price_pnl_usdt"] for trade in own), "funding_usdt":symbol_funding[symbol],
            "realized_pnl_usdt":symbol_cash[symbol]})
    return {"mode":settings.volium_mode, "symbols":symbols, "initial_equity":initial,
        "first_execution_open_utc":clocks[0].isoformat(), "last_execution_close_utc":(clocks[-1]+duration).isoformat(),
        "execution_minutes":len(clocks)*SECONDS[execution_timeframe]//60, "execution_timeframe":execution_timeframe,
        "execution_bars":len(clocks), "coarse_execution_proxy":execution_timeframe != "1m", "final_realized_equity":equity,
        "admission_adapter":getattr(admission, "policy", {}), "engine_version":"v5_cost_aware",
        "positive_signal_schedule_applied":signal_schedule is not None,
        "daily_blocked_signal_checks_scope":"scheduled_positive_boundaries" if signal_schedule is not None else "all_eligible_boundaries",
        "historical_contract_constraints_known":False,
        "exchange_liquidations_reconstructed":False,
        "fees_total_usdt":fees_total, "slippage_total_usdt":slippage_total,
        "gross_closed_price_pnl_usdt":gross_price_total,
        "open_entry_cost_usdt":sum(order["entry_fee"]+order["entry_slippage"] for order in active.values()),
        "admission_rejections_by_reason":admission_rejections,
        "net_pnl_usdt":equity-initial, "return_percent":(equity/initial-1)*100,
        "unrealized_pnl_usdt":unrealized, "marked_equity":marked_equity,
        "max_realized_drawdown_percent":max_realized_dd, "max_marked_drawdown_percent":max_marked_dd,
        "trades_count":len(trades), "wins":wins, "win_rate":wins/len(trades)*100 if trades else None,
        "profit_factor":profits/losses if losses else None, "funding_total_usdt":funding_total,
        "peak_active_orders_positions":peak_active, "peak_reserved_margin_usdt":peak_margin,
        "peak_margin_utilization_percent":peak_margin_utilization,
        "unfinished_positions":unfinished_positions, "unfinished_limits":len(active)-unfinished_positions,
        "unfilled_signals":counts["accepted_signals"]-len(trades)-unfinished_positions,
        "active_orders_positions":[{"symbol":symbol,"id":order["setup"].id,"filled":order["filled"],
            "notional":order["notional"],"leverage":order["leverage"],"margin_reserved_usdt":reservation(order),
            "entry_fee_usdt":order["entry_fee"], "entry_slippage_usdt":order["entry_slippage"],
            "cash_risk_budget_usdt":order["risk_budget"], "economics":order["setup"].economics,
            "signal_time":order["setup"].timestamp.isoformat(),"entry_time":order.get("entry_time"),
            "funding_usdt":order["funding"]} for symbol,order in active.items()],
        "missing_execution_minutes_by_symbol":{symbol:int((positions[symbol]<0).sum())*SECONDS[execution_timeframe]//60 for symbol in symbols},
        "missing_execution_bars_by_symbol":{symbol:int((positions[symbol]<0).sum()) for symbol in symbols},
        **counts, "symbol_results":symbol_results, "daily_close":list(daily_close.values()), "trades":trades}

