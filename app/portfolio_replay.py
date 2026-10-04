"""Causal shared-account replay using a frozen public-data cache only."""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Settings
from app.replay import SECONDS, required_timeframes
from app.strategy.volium import analyze_volium_from_df, in_volium_session


def _utc(value):
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _frame(frame):
    result = frame.copy()
    result.index = pd.to_datetime(result.index, utc=True)
    result = result.sort_index()
    fields = ["open", "high", "low", "close"]
    if result.empty or result.index.has_duplicates or not set(fields).issubset(result):
        raise ValueError("Portfolio requires nonempty unique UTC OHLC frames")
    values = result[fields].to_numpy(dtype=float)
    opens, highs, lows, closes = values.T
    if not np.isfinite(values).all() or not ((lows > 0) & (lows <= np.minimum(opens, closes))
            & (highs >= np.maximum(opens, closes)) & (highs >= lows)).all():
        raise ValueError("Portfolio OHLC is invalid")
    return result


def portfolio_replay(frames_by_symbol, settings, *, symbols=None, start_at=None, end_at=None,
                     execution_frames=None, funding_rates=None, signal_provider=analyze_volium_from_df,
                     execution_timeframe="1m", position_limits=None, signal_schedule=None):
    """One shared account and signals at their original M5/M1 closes.

    Each execution bar processes funding/exits for ALL symbols before admitting new
    orders in supplied universe order. Pending orders reserve margin and slots.
    Risk uses realized account equity; margin is capped at 90% of that equity.
    New size is capped by available shared margin, rather than borrowing it twice.
    M1 is the default. Explicit M5 execution is a coarser OHLC proxy for research,
    not synthesized M1 data; scalp M1 confirmation cannot use coarser execution.
    Optional current contract rules are a declared research proxy for historical
    lot/tick/max-volume constraints. Default None preserves the unrounded model.
    An optional externally precomputed positive-signal schedule skips provider
    calls at confirmed closes known to yield None; all price/funding clocks stay
    complete. Its causal provenance belongs to the supplying research runner.
    """
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
    position_limits = position_limits or {}
    if not set(position_limits).issubset(symbols):
        raise ValueError("Contract constraints must belong to portfolio symbols")
    for symbol, contract in position_limits.items():
        required = ("contractSize", "volUnit", "minVol", "maxVol", "priceUnit")
        if any(key not in contract or not np.isfinite(float(contract[key])) or float(contract[key]) <= 0 for key in required):
            raise ValueError(f"Invalid contract constraint snapshot for {symbol}")
        if float(contract["minVol"]) > float(contract["maxVol"]):
            raise ValueError("Contract minimum volume exceeds maximum")
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
    counts.update({"downsized_for_contract_limits": 0, "rejected_contract_limits": 0,
                   "rounded_contract_prices": 0})
    funding_total = 0.0
    symbol_funding = {symbol: 0.0 for symbol in symbols}
    symbol_signals = {symbol: 0 for symbol in symbols}
    symbol_accepted = {symbol: 0 for symbol in symbols}
    peak_active = 0
    peak_margin = peak_margin_utilization = 0.0
    daily_pnl, day, daily_opening_equity = 0.0, None, initial
    costs = 2 * (settings.paper_fee_bps + settings.paper_slippage_bps) / 10000
    leverage = float(settings.default_leverage)
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
                    equity += cash
                    daily_pnl += cash
                    counts["funding_events_charged"] += 1
            if stop_hit or target_hit:
                exit_price = setup.stop_loss if stop_hit else setup.take_profits[0]
                if stop_hit:
                    exit_price = min(exit_price, open_price) if long else max(exit_price, open_price)
                pnl = order["notional"] * (sign * (exit_price - setup.entry) / setup.entry - costs)
                equity += pnl
                daily_pnl += pnl
                trades.append({"symbol": symbol, "id": setup.id, "signal_time":setup.timestamp.isoformat(),
                    "entry_time":order["entry_time"], "exit_time":closed.isoformat(), "direction":setup.direction,
                    "entry":setup.entry, "stop":setup.stop_loss, "target":setup.take_profits[0], "exit":float(exit_price),
                    "notional":order["notional"], "margin_reserved_usdt":order["notional"]/leverage,
                    "price_pnl_usdt":pnl, "funding_usdt":order["funding"], "pnl_usdt":pnl+order["funding"],
                    "status":"SL_HIT" if stop_hit else "TP1_HIT"})
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
            known = {tf: frame.iloc[:close_indices[symbol][tf].searchsorted(closed, side="right")].tail(settings.volium_context_lookback+30)
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
            if symbol in position_limits:
                from app.exchange.client import round_bracket_prices
                try:
                    entry, stop, take = round_bracket_prices(position_limits[symbol], setup.direction,
                                                            setup.entry, setup.stop_loss, setup.take_profits[0])
                except ValueError:
                    counts["rejected_contract_limits"] += 1
                    continue
                prices = (float(entry), float(stop), float(take))
                counts["rounded_contract_prices"] += int(prices != (setup.entry, setup.stop_loss, setup.take_profits[0]))
                setup = setup.model_copy(update={"entry": prices[0], "stop_loss": prices[1],
                                                 "take_profits": [prices[2]],
                                                 "risk_reward": abs(prices[2]-prices[0])/abs(prices[0]-prices[1])})
            distance = abs(setup.entry-setup.stop_loss) / setup.entry
            long = setup.direction == "LONG"
            valid = (0 < setup.stop_loss < setup.entry < setup.take_profits[0] if long
                     else 0 < setup.take_profits[0] < setup.entry < setup.stop_loss)
            if not valid or not np.isfinite(distance) or distance <= 0:
                counts["rejected_geometry"] += 1
                continue
            reserved = sum(order["notional"] / leverage for order in active.values())
            available = max(0.0, equity * 0.9 - reserved)
            if available <= 1e-9:
                counts["rejected_margin"] += 1
                continue
            risk_notional = equity * settings.risk_per_trade_percent / 100 / distance
            notional = min(risk_notional, available * leverage)
            counts["downsized_for_margin"] += int(notional + 1e-9 < risk_notional)
            if symbol in position_limits:
                from app.exchange.client import contracts_for_notional
                contract = position_limits[symbol]
                maximum = min(Decimal(str(contract["maxVol"])), Decimal(str(contract.get("limitMaxVol", contract["maxVol"]))))
                maximum_notional = maximum * Decimal(str(contract["contractSize"])) * Decimal(str(setup.entry))
                bounded = min(Decimal(str(notional)), maximum_notional)
                try:
                    volume = contracts_for_notional(contract, bounded, setup.entry)
                except ValueError:
                    counts["rejected_contract_limits"] += 1
                    continue
                quantized = float(volume * Decimal(str(contract["contractSize"])) * Decimal(str(setup.entry)))
                counts["downsized_for_contract_limits"] += int(quantized + 1e-9 < notional)
                notional = quantized
            active[symbol] = {"setup":setup, "notional":notional, "filled":False, "funding":0.0}
            accepted_ids.add((symbol,setup.id))
            counts["accepted_signals"] += 1
            symbol_accepted[symbol] += 1
        reserved = sum(order["notional"] / leverage for order in active.values())
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
            "realized_pnl_usdt":sum(trade["price_pnl_usdt"] for trade in own)+symbol_funding[symbol]})
    return {"mode":settings.volium_mode, "symbols":symbols, "initial_equity":initial,
        "first_execution_open_utc":clocks[0].isoformat(), "last_execution_close_utc":(clocks[-1]+duration).isoformat(),
        "execution_minutes":len(clocks)*SECONDS[execution_timeframe]//60, "execution_timeframe":execution_timeframe,
        "execution_bars":len(clocks), "coarse_execution_proxy":execution_timeframe != "1m", "final_realized_equity":equity,
        "contract_constraint_snapshot_applied":position_limits,
        "positive_signal_schedule_applied":signal_schedule is not None,
        "daily_blocked_signal_checks_scope":"scheduled_positive_boundaries" if signal_schedule is not None else "all_eligible_boundaries",
        "historical_contract_constraints_known":False if position_limits else None,
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
            "notional":order["notional"],"margin_reserved_usdt":order["notional"]/leverage,
            "signal_time":order["setup"].timestamp.isoformat(),"entry_time":order.get("entry_time"),
            "funding_usdt":order["funding"]} for symbol,order in active.items()],
        "missing_execution_minutes_by_symbol":{symbol:int((positions[symbol]<0).sum())*SECONDS[execution_timeframe]//60 for symbol in symbols},
        "missing_execution_bars_by_symbol":{symbol:int((positions[symbol]<0).sum()) for symbol in symbols},
        **counts, "symbol_results":symbol_results, "daily_close":list(daily_close.values()), "trades":trades}


def run_cache(cache, selection_path, output_directory, days=30):
    """Validate snapshot hashes and run both modes without constructing any API client."""
    cache, output_directory = Path(cache), Path(output_directory)
    manifest = json.loads((cache / "snapshot.json").read_text(encoding="utf-8"))
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    symbols = [row["symbol"] for row in selection["selected"]]
    if len(symbols) != 5 or len(set(symbols)) != 5:
        raise ValueError("Frozen selection must contain exactly five unique symbols")
    end = _utc(manifest["server_snapshot_utc"])
    start = end - pd.Timedelta(days=days)
    frames, rates, source_hashes = {}, {}, {}
    for symbol in symbols:
        frames[symbol] = {}
        for tf in ("1d", "1h", "5m", "1m"):
            path = cache / f"{symbol}_{tf}.csv"
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["frames"][symbol][tf]["sha256"]:
                raise ValueError(f"Modified frozen public cache: {path.name}")
            source_hashes[path.name] = manifest["frames"][symbol][tf]["sha256"]
            frame = pd.read_csv(path, index_col=0)
            frame.index = pd.to_datetime(frame.index, utc=True)
            frames[symbol][tf] = frame.loc[frame.index + pd.Timedelta(seconds=SECONDS[tf]) <= end]
        path = cache / f"{symbol}_funding.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["funding"][symbol]["sha256"]:
            raise ValueError(f"Modified frozen public funding cache: {path.name}")
        source_hashes[path.name] = manifest["funding"][symbol]["sha256"]
        series_frame = pd.read_csv(path, index_col=0)
        series_frame.index = pd.to_datetime(series_frame.index, utc=True)
        rates[symbol] = series_frame["funding_rate"].loc[series_frame.index <= end]
    results = []
    for mode in ("intraday", "scalp"):
        settings = Settings(_env_file=None, mexc_api_key="", mexc_api_secret="", auto_execution=False,
            volium_mode=mode, account_balance_usdt=1000, risk_per_trade_percent=.5, max_open_setups=3,
            daily_loss_limit_percent=2, default_leverage=3, pending_order_max_age_minutes=30,
            paper_fee_bps=5, paper_slippage_bps=2, volium_session_enabled=True,
            volium_session_clock="fixed_utc3", volium_sessions_utc3=[("10:00","12:00"),("16:30","18:00")],
            volium_swing_lookback=2, volium_context_lookback=80, volium_sweep_max_age_bars=12,
            volium_reaction_max_bars=3, volium_min_body_ratio=.6, volium_require_daily_origin_sweep=True,
            volium_stop_buffer_bps=2, volium_scalp_stop_buffer_bps=10, volium_active_trend_min_atr=1)
        print(f"Shared-account {mode}: {days} days, M1 execution, {', '.join(symbols)}", flush=True)
        results.append(portfolio_replay(frames, settings, symbols=symbols, start_at=start, end_at=end, funding_rates=rates))
        print(json.dumps({key:results[-1][key] for key in ("mode","trades_count","final_realized_equity","marked_equity","peak_active_orders_positions")}), flush=True)
    code_files = {"portfolio_replay":Path(__file__),"strategy":Path(__file__).parent/"strategy/volium.py",
                  "config":Path(__file__).parent/"config.py"}
    report = {"source":"Frozen MEXC public Futures OHLC and funding cache", "snapshot_utc":end.isoformat(),
        "code_sha256":{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in code_files.items()},
        "source_data_sha256":source_hashes,
        "requested_start_utc":start.isoformat(), "requested_days":days, "universe_selection":selection,
        "assumptions":{"shared_initial_equity_usdt":1000,"risk_per_trade_percent":.5,"max_active_slots":3,
            "daily_loss_limit_percent":2,"daily_threshold":"frozen UTC day opening realized equity",
            "leverage":3,"margin_reservation":"pending+filled notional/3; at most 90% realized equity on admission",
            "simultaneous_signal_priority":"frozen selected universe order; all exits/funding settle before new orders",
            "margin_sizing":"remaining shared margin caps requested risk size","execution":"M1",
            "deduplication":"accepted ideas are consumed; rejected ideas may retry on a later confirmation",
            "signal_counts":"unique observed ideas; rejection counts are attempts, not unique ideas",
            "missing_execution_data":"abort if any symbol lacks any requested complete M1 minute",
            "signal_timeframes":{"intraday":"M5","scalp":"M1"},"fee_bps_per_side":5,"slippage_bps_per_side":2,
            "funding_mark_price_proxy":"last known M1 close","no_favorable_credit_on_ambiguous_funding_bar":True,
            "stop_priority_when_both_touched":True,"no_tp_on_uncertain_limit_fill_bar":True,
            "partial_fills":"not modeled","liquidations_and_spread":"not modeled",
            "selection_bias":"current frozen five applied to past history; historical selection bias remains"},
        "results":results}
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "portfolio_backtest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines = ["# Общий торговый счёт: проверка портфеля", "",
        f"Зафиксированный конец данных: {end.isoformat()}. Окно: последние {days} дней, начало {start.isoformat()}.", "",
        f"Порядок пар: {', '.join(symbols)}. Каждый режим отдельно стартует с **общих 1 000 USDT**.", "",
        "| Режим | Сделок | Сигналов / принятых | Итог реализованный | Итог с открытыми позициями | Доходность | Макс. просадка по M1 закрытиям | Funding |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        lines.append(f"| {result['mode']} | {result['trades_count']} | {result['signals']} / {result['accepted_signals']} | {result['final_realized_equity']:.2f} | {result['marked_equity']:.2f} | {result['return_percent']:.2f}% | {result['max_marked_drawdown_percent']:.2f}% | {result['funding_total_usdt']:.4f} |")
    lines.extend(["", "Это отдельное причинное воспроизведение одного счёта; индивидуальные доходности пяти счетов не складывались. Риск заявки — до 0.5% реализованного общего капитала, плечо 3, одновременно не больше трёх заявок/позиций. Одна активная идея на пару. Лимитные заявки резервируют маржу и место до заполнения либо истечения 30 минут. Их общий резерв с открытыми позициями ограничен 90% капитала при приёме заявки; недостаток доступной маржи уменьшает номинал следующей заявки.", "",
        "Все выходы и funding пяти пар на одной M1-границе учитываются до новых заявок. Одновременные сигналы получают приоритет в зафиксированном порядке списка выше; он не сортируется по будущей доходности. Принятая идея с тем же ID больше не торгуется; отклонённая из-за мест/маржи может быть повторно рассмотрена при следующем закрытом подтверждении. Число сигналов считает уникальные наблюдаемые идеи, число отказов — попытки. Дневной убыток 2% считается от реализованного капитала в начале дня UTC; после порога новые заявки запрещены, действующие позиции и заявки продолжают исполняться. При отсутствии хотя бы одной требуемой M1-свечи любой пары тест прерывается.", "",
        "Сигнал intraday вычисляется только после закрытия M5, scalp — после M1; стратегия видит лишь уже закрытые H1/D1/M5. Заполнение доступно только с последующей M1-свечи. Стоп имеет приоритет при одновременном касании цели; выгодная цель не засчитывается на неоднозначной свече лимитного заполнения. Комиссия 5 bps и проскальзывание 2 bps на сторону списываются при закрытии. Реальные публичные ставки funding применены с ценой последнего известного M1-закрытия как прокси; неоднозначное выгодное поступление исключается.", ""])
    for result in results:
        lines.append(f"{result['mode']}: максимум активных мест {result['peak_active_orders_positions']}; максимум зарезервированной маржи {result['peak_reserved_margin_usdt']:.2f} USDT; отклонено по местам {result['rejected_slots']}, по марже {result['rejected_margin']}; уменьшено по марже {result['downsized_for_margin']}; незаполненных сигналов {result['unfilled_signals']}; незавершённых позиций {result['unfinished_positions']}, лимитных заявок {result['unfinished_limits']}.")
        lines.append("")
    lines.extend(["**Ограничения:** окно только 30 дней. Пять пар выбраны текущим селектором и применены к прошлому; исторический отбор и survivorship bias не устранены. Режимы не объединены в одновременно торгующую систему. Частичные заполнения, очередь лимитных заявок, точный bid/ask, ликвидация и историческая fair/mark price не моделируются. Маржа фиксируется по номиналу входа; при последующих потерях её отношение к капиталу может превысить первоначальный порог. Просадка измерена на закрытиях M1, не внутри свечи. Этот результат относится к выбранной автоматизации визуальных правил автора, а не доказывает прибыльность ручной стратегии.", "",
        "Полные сделки, funding, дневной капитал и результаты каждой пары записаны в `portfolio_backtest.json`. Источник стратегии: https://www.youtube.com/watch?v=jYvt1hSTPxc.", ""])
    (output_directory / "portfolio_report.md").write_text("\n".join(lines),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=".cache/suite")
    parser.add_argument("--selection", default=".cache/universe.json")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.days <= 30:
        parser.error("--days must be between 1 and 30 (cached M1 coverage)")
    run_cache(args.cache, args.selection, args.out_dir, args.days)
