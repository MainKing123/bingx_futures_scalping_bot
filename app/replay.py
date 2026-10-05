"""Reproducible public-data replay. Never imports the order executor or API keys."""
from __future__ import annotations

import argparse
import asyncio
import json
from math import isfinite
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.config import Settings
from app.exchange.client import MEXCClient
from app.strategy.volium import analyze_volium_from_df, in_volium_session
from app.universe import UniverseSelector

SECONDS = {"1m":60, "5m":300, "1h":3600, "4h":14400, "1d":86400, "1w":604800}


def required_timeframes(settings):
    if settings.volium_mode == "intraday":
        return ("1d","1h","5m")
    if settings.volium_mode == "scalp":
        return ("1h","5m","1m")
    return ("1w","4h") if settings.volium_swing_context == "1w" else ("1d","1h")


def replay(symbol, frames, settings, start_at=None, signal_provider=analyze_volium_from_df, execution_frame=None, funding_rates=None):
    """Causal replay; optional execution_frame is finer 1m OPEN-time OHLC.

    Signals are evaluated at strategy entry-frame closes only. Without finer
    bars, an intrabar expiry never assumes an uncertain limit fill succeeded.
    Risk sizing compounds realized equity; each UTC day's loss threshold is
    frozen at that day's opening equity. Fees/slippage are round-trip costs.
    """
    tfs = required_timeframes(settings)
    if not set(tfs).issubset(frames):
        raise ValueError(f"Missing strategy frames: {sorted(set(tfs) - set(frames))}")
    frames = {tf: frame if frame.index.is_monotonic_increasing else frame.sort_index()
              for tf, frame in frames.items()}
    entry_tf = tfs[-1]
    execution_tf = "1m" if execution_frame is not None else entry_tf
    df = execution_frame if execution_frame is not None else frames[entry_tf]
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    signal_times = set(frames[entry_tf].index + pd.Timedelta(seconds=SECONDS[entry_tf]))
    close_indices = {tf: frame.index + pd.Timedelta(seconds=SECONDS[tf]) for tf, frame in frames.items()}
    execution_closes = df.index + pd.Timedelta(seconds=SECONDS[execution_tf])
    rates = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC")) if funding_rates is None else funding_rates.copy()
    rates.index = pd.to_datetime(rates.index, utc=True)
    rates = rates.loc[~rates.index.duplicated(keep="last")].sort_index().astype(float)
    if not all(isfinite(value) for value in rates):
        raise ValueError("Funding rates must be finite")
    funding_cursor, funding_total, funding_events, skipped_favorable = 0, 0.0, 0, 0
    initial = settings.account_balance_usdt
    equity, peak, max_dd = initial, initial, 0.0
    marked_equity, marked_peak, max_marked_dd, unrealized = initial, initial, 0.0, 0.0
    pending = None
    filled = False
    seen = set()
    trades = []
    signals, expired_limits, ambiguous_expiry_limits = 0, 0, 0
    daily_pnl, day, daily_opening_equity = 0.0, None, initial
    for ts, row in df.iterrows():
        opened = ts.to_pydatetime()
        closed = opened + timedelta(seconds=SECONDS[execution_tf])
        if start_at and opened < start_at:
            continue
        if closed.date() != day:
            day, daily_pnl, daily_opening_equity = closed.date(), 0.0, equity
        events = []
        while funding_cursor < len(rates) and rates.index[funding_cursor] <= pd.Timestamp(closed):
            events.append((rates.index[funding_cursor], float(rates.iloc[funding_cursor])))
            funding_cursor += 1
        if pending is not None:
            setup = pending["setup"]
            high, low = float(row.high), float(row.low)
            filled_this_bar = False
            opened_through_limit = (float(row.open) <= setup.entry if setup.direction == "LONG" else float(row.open) >= setup.entry)
            if not filled:
                expiry = setup.timestamp + timedelta(minutes=settings.pending_order_max_age_minutes)
                limit_touched = low <= setup.entry <= high
                if opened >= expiry:
                    pending = None
                    expired_limits += 1
                elif opened_through_limit or (limit_touched and closed <= expiry):
                    filled = filled_this_bar = True
                    pending["entry_time"] = opened.isoformat()
                    pending["entry_at"] = opened
                elif closed > expiry:
                    # Whole-bar ranges cannot tell whether a limit touch occurred
                    # before or after a TTL that falls inside this bar.
                    ambiguous_expiry_limits += int(limit_touched)
                    expired_limits += 1
                    pending = None
            if pending is not None and filled:
                sl = low <= setup.stop_loss if setup.direction == "LONG" else high >= setup.stop_loss
                tp = high >= setup.take_profits[0] if setup.direction == "LONG" else low <= setup.take_profits[0]
                if filled_this_bar and not opened_through_limit:
                    # A favorable high/low may precede the limit fill in this bar.
                    tp = False
                sign = 1 if setup.direction == "LONG" else -1
                exit_at_open = ((sl and (float(row.open) <= setup.stop_loss if sign == 1 else float(row.open) >= setup.stop_loss))
                                or (tp and (float(row.open) >= setup.take_profits[0] if sign == 1 else float(row.open) <= setup.take_profits[0])))
                for settled_at, rate in events:
                    if settled_at < pd.Timestamp(pending["entry_at"]) or (exit_at_open and settled_at > pd.Timestamp(opened)):
                        continue
                    mark_i = execution_closes.searchsorted(settled_at, side="right") - 1
                    mark = float(df.close.iloc[mark_i]) if mark_i >= 0 else float(row.open)
                    cash = -sign * rate * (pending["notional"] / setup.entry) * mark
                    known_held = not (sl or tp) and (not filled_this_bar or (opened_through_limit and settled_at > pd.Timestamp(opened)))
                    if cash > 0 and not known_held:
                        skipped_favorable += 1
                        continue
                    if cash:
                        pending["funding"] += cash
                        funding_total += cash
                        equity += cash
                        daily_pnl += cash
                        funding_events += 1
                if sl or tp:
                    exit_price = setup.stop_loss if sl else setup.take_profits[0]
                    if sl:
                        exit_price = min(exit_price, float(row.open)) if setup.direction == "LONG" else max(exit_price, float(row.open))
                    gross = sign*(exit_price-setup.entry)/setup.entry
                    costs = 2*(settings.paper_fee_bps+settings.paper_slippage_bps)/10000
                    pnl = pending["notional"]*(gross-costs)
                    equity += pnl
                    daily_pnl += pnl
                    trades.append({"id":setup.id, "signal_time":setup.timestamp.isoformat(),
                        "entry_time":pending["entry_time"], "exit_time":closed.isoformat(),
                        "direction":setup.direction, "entry":setup.entry, "stop":setup.stop_loss,
                        "target":setup.take_profits[0], "exit":exit_price, "notional":pending["notional"],
                        "price_pnl_usdt":pnl, "funding_usdt":pending["funding"],
                        "pnl_usdt":pnl+pending["funding"], "status":"SL_HIT" if sl else "TP1_HIT"})
                    pending, filled = None, False
        peak = max(peak,equity)
        max_dd = max(max_dd,(peak-equity)/peak*100)
        unrealized = 0.0
        if pending is not None and filled:
            setup = pending["setup"]
            sign = 1 if setup.direction == "LONG" else -1
            unrealized = sign * pending["notional"] * (float(row.close) - setup.entry) / setup.entry
        marked_equity = equity + unrealized
        marked_peak = max(marked_peak,marked_equity)
        max_marked_dd = max(max_marked_dd,(marked_peak-marked_equity)/marked_peak*100)
        if pending is not None or equity <= 0 or daily_pnl <= -daily_opening_equity*settings.daily_loss_limit_percent/100:
            continue
        if pd.Timestamp(closed) not in signal_times:
            continue
        if settings.volium_mode != "swing" and settings.volium_session_enabled and not in_volium_session(closed, settings):
            continue
        known = {tf:frame.iloc[:close_indices[tf].searchsorted(pd.Timestamp(closed), side="right")].tail(settings.volium_context_lookback+30)
                 for tf,frame in frames.items()}
        setup = signal_provider(symbol=symbol, frames=known, settings=settings, mode=settings.volium_mode, now=closed)
        if setup is None or setup.id in seen:
            continue
        seen.add(setup.id)
        signals += 1
        distance = abs(setup.entry-setup.stop_loss)/setup.entry
        if distance <= 0:
            continue
        notional = min(equity*settings.risk_per_trade_percent/100/distance, equity*settings.default_leverage*0.9)
        pending = {"setup":setup,"notional":notional,"funding":0.0}
        filled = False
    profits = sum(max(t["pnl_usdt"],0) for t in trades)
    losses = -sum(min(t["pnl_usdt"],0) for t in trades)
    wins = sum(t["pnl_usdt"]>0 for t in trades)
    return {"symbol":symbol,"mode":settings.volium_mode,
        "initial_equity":initial,"final_realized_equity":equity,
        "net_pnl_usdt":equity-initial,"return_percent":(equity/initial-1)*100,
        "funding_total_usdt":funding_total,"funding_events_charged":funding_events,
        "funding_events_skipped_favorable":skipped_favorable,
        "unrealized_pnl_usdt":unrealized,"marked_equity":marked_equity,
        "max_marked_drawdown_percent":max_marked_dd,
        "signals":signals,"trades_count":len(trades),"wins":wins,
        "execution_timeframe":execution_tf,"expired_limits":expired_limits,
        "ambiguous_expiry_limits":ambiguous_expiry_limits,
        "unfilled_signals":signals-len(trades)-int(bool(pending and filled)),
        "win_rate":wins/len(trades)*100 if trades else None,
        "profit_factor":profits/losses if losses else None,
        "max_realized_drawdown_percent":max_dd,
        "unfinished_position":bool(pending and filled),"unfinished_limit":bool(pending and not filled),
        "trades":trades}


async def run(args):
    # Explicit empty credentials: replay can access public market data only.
    settings = Settings(_env_file=None, mexc_api_key="", mexc_api_secret="", auto_execution=False,
        volium_mode=args.mode, volium_swing_context=args.swing_context)
    client = MEXCClient(settings)
    results = []
    selector = None
    try:
        selection = None
        if args.symbol:
            symbols = list(dict.fromkeys(args.symbol))
        else:
            selector = UniverseSelector(client, settings)
            selected = await selector.select()
            symbols = [row["symbol"] for row in selected]
            selection = selector.snapshot
        server_ms = await client.get_server_time()
        server_now = datetime.fromtimestamp(server_ms/1000,timezone.utc)
        start_at = server_now - timedelta(days=args.days)
        cache = Path(args.cache)
        cache.mkdir(parents=True,exist_ok=True)
        for symbol in symbols:
            frames = {}
            for tf in required_timeframes(settings):
                limit = min(100000,int(args.days*86400/SECONDS[tf])+settings.volium_context_lookback+50)
                frame = await client.get_klines(symbol,tf,limit=limit,end_time=server_ms)
                if frame.empty:
                    raise RuntimeError(f"No public candles for {symbol} {tf}")
                frames[tf] = frame
                frame.to_csv(cache / f"{symbol}_{tf}.csv")
            print(f"Replaying {symbol} {args.mode}: {len(frames[required_timeframes(settings)[-1]])} entry candles",flush=True)
            funding = await client.get_funding_history(symbol, int(start_at.timestamp()*1000))
            funding = funding.loc[funding.index <= pd.Timestamp(server_now)]
            funding.to_csv(cache / f"{symbol}_funding.csv")
            results.append(replay(symbol,frames,settings,start_at=start_at,funding_rates=funding))
        report = {"source":"MEXC public Futures OHLC","fetched_at":server_now.isoformat(),
            "symbols":symbols,"universe_selection":selection,
            "historical_universe_policy":"Current fixed selection applied to history; selection bias is not removed",
            "requested_days":args.days,"strategy_video":"https://www.youtube.com/watch?v=jYvt1hSTPxc",
            "assumptions":{"equity_per_symbol":settings.account_balance_usdt,"risk_percent":settings.risk_per_trade_percent,
                "leverage_cap":settings.default_leverage,"fee_bps_per_side":settings.paper_fee_bps,
                "slippage_bps_per_side":settings.paper_slippage_bps,"session_utc3":settings.volium_sessions_utc3,
                "stop_priority_if_both_touched":True,"no_tp_on_uncertain_fill_bar":True,
                "unrealized_pnl_reported_separately":True,"independent_symbol_accounts":True,
                "actual_funding_rates":True,"funding_mark_price_proxy":"last known execution candle close",
                "execution_resolution":"strategy entry timeframe; use comprehensive suite for M1 comparisons"},
            "results":results}
        out = Path(args.out)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps([{k:v for k,v in r.items() if k!="trades"} for r in results],ensure_ascii=False,indent=2))
    finally:
        if selector is not None:
            await selector.close()
        await client.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["intraday","scalp","swing"],default="intraday")
    parser.add_argument("--swing-context",choices=["1d","1w"],default="1d")
    parser.add_argument("--symbol",action="append",default=None)
    parser.add_argument("--days",type=int,default=30)
    parser.add_argument("--out",default="outputs/replay.json")
    parser.add_argument("--cache",default=".cache/candles")
    args=parser.parse_args()
    if not 1 <= args.days <= 365:
        parser.error("--days must be between 1 and 365")
    asyncio.run(run(args))
