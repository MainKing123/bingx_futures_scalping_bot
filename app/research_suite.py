"""Finite preregistered VOLIUM experiments. Reads verified public CSVs only.

Register a plan, run train, run validation/freeze, then explicitly open final.
No strategy selection is allowed to inspect final-test returns.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import Settings
from app.portfolio_replay import portfolio_replay
from app.replay import SECONDS, required_timeframes
from app.schemas.setup import TradeSetup
from app.strategy.volium import analyze_volium_from_df, in_volium_session

CODE_FILES = ("research_suite.py", "portfolio_replay.py", "replay.py", "config.py",
              "strategy/volium.py", "strategy/volium_v2.py", "schemas/setup.py", "exchange/client.py")
STAGES = ("train", "validation", "test")
POLICY = {"train_min_trades": 10, "validation_min_trades": 3, "test_min_trades": 3,
          "shortlist_limit": 3, "small_sample_below": 30,
          "selection_requires_no_unfinished_positions": True,
          "ranking": ["profit_factor", "net_return", "train_profit_factor", "stable_id"],
          "baseline_costs_bps_per_side": [5, 2], "stress_costs_bps_per_side": [10, 5]}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def candidate_grid():
    result = []
    for atr, body, prior, bars in itertools.product((.8, 1.2, 1.6), (.6, .7), (1., 1.5), (2, 3)):
        params = {"context_mode": "preexisting_target", "correction_target_mode": "leg_origin",
                  "reaction_atr_period": 14, "reaction_min_atr": atr,
                  "reaction_min_body_ratio": body, "reaction_min_prior_body_ratio": prior,
                  "reaction_max_bars": bars}
        result.append({"id": "v2-" + digest(params)[:16], "category": "candidate", "parameters": params})
    for atr, prior in itertools.product((.8, 1.2, 1.6), (1., 1.5)):
        params = {"context_mode": "legacy_context", "correction_target_mode": "leg_origin",
                  "reaction_atr_period": 14, "reaction_min_atr": atr,
                  "reaction_min_body_ratio": .6, "reaction_min_prior_body_ratio": prior,
                  "reaction_max_bars": 3}
        result.append({"id": "ablation-" + digest(params)[:16], "category": "semantic_ablation", "parameters": params})
    result.append({"id": "v1-control", "category": "control", "parameters": None})
    return result


def chronological_folds(start, end, execution_timeframe="1m"):
    seconds = SECONDS[execution_timeframe]
    frequency = f"{seconds}s"
    start, end = utc(start).ceil(frequency), utc(end).floor(frequency)
    bars = int((end-start).total_seconds() // seconds)
    sizes = [bars*6//10, bars*2//10]
    sizes.append(bars-sum(sizes))
    if min(sizes) < 1:
        raise ValueError("Research requires three nonempty chronological folds")
    folds, cursor = {}, start
    for stage, size in zip(STAGES, sizes):
        finish = cursor + pd.Timedelta(seconds=size*seconds)
        folds[stage] = {"start_utc": cursor.isoformat(), "end_utc": finish.isoformat(), "execution_bars": size}
        cursor = finish
    return folds


def validate_folds(folds, resolution):
    duration = pd.Timedelta(seconds=SECONDS[resolution])
    previous = None
    result = {}
    for stage in STAGES:
        start, end = utc(folds[stage]["start_utc"]), utc(folds[stage]["end_utc"])
        if start >= end or start.value % duration.value or end.value % duration.value:
            raise ValueError("Fold boundaries must align to the declared execution resolution")
        if previous is not None and start != previous:
            raise ValueError("Research folds must be contiguous and chronological")
        result[stage] = {"start_utc": start.isoformat(), "end_utc": end.isoformat(),
                         "execution_bars": int((end-start)/duration)}
        previous = end
    return result


def settings_for(mode, *, stress=False):
    return Settings(_env_file=None, mexc_api_key="", mexc_api_secret="", auto_execution=False,
                    account_balance_usdt=1000, risk_per_trade_percent=.5, default_leverage=3,
                    daily_loss_limit_percent=2, max_open_setups=3, pending_order_max_age_minutes=30,
                    volium_mode=mode, volium_session_enabled=True, volium_session_clock="fixed_utc3",
                    volium_sessions_utc3=[("10:00", "12:00"), ("16:30", "18:00")],
                    paper_fee_bps=10 if stress else 5, paper_slippage_bps=5 if stress else 2)


def source_paths(cache, manifest, symbols, modes, resolution):
    required = {resolution}
    for mode in modes:
        required.update(required_timeframes(settings_for(mode)))
    paths = {}
    for symbol in symbols:
        for tf in sorted(required):
            metadata = manifest["frames"][symbol][tf]
            name = metadata.get("filename", f"{symbol}_{tf}.csv")
            paths[name] = (Path(cache)/name, metadata["sha256"])
        metadata = manifest["funding"][symbol]
        if metadata.get("status", "available") != "available":
            raise ValueError(f"Actual same-venue funding required: {symbol}")
        name = metadata.get("filename", f"{symbol}_funding.csv")
        paths[name] = (Path(cache)/name, metadata["sha256"])
    return paths


def read_dataset(plan, *, end_at=None):
    cache = Path(plan["cache_path"])
    manifest = json.loads((cache/"snapshot.json").read_text(encoding="utf-8"))
    if file_hash(cache/"snapshot.json") != plan["data_manifest_sha256"]:
        raise ValueError("Data manifest changed after registration")
    symbols = list(dict.fromkeys(symbol for market in plan["markets"] for symbol in market["symbols"]))
    paths = source_paths(cache, manifest, symbols, plan["modes"], plan["execution_resolution"])
    for name, (path, expected) in paths.items():
        if file_hash(path) != expected or expected != plan["data_sha256"][name]:
            raise ValueError(f"Public data changed after registration: {name}")
    required = {plan["execution_resolution"]}
    for mode in plan["modes"]:
        required.update(required_timeframes(settings_for(mode)))
    frames, funding = {}, {}
    cutoff = utc(end_at or plan["folds"]["test"]["end_utc"])
    for symbol in symbols:
        frames[symbol] = {}
        for tf in required:
            name = manifest["frames"][symbol][tf].get("filename", f"{symbol}_{tf}.csv")
            frame = pd.read_csv(cache/name, index_col=0)
            frame.index = pd.to_datetime(frame.index, utc=True)
            if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
                raise ValueError(f"Invalid public OHLC chronology: {name}")
            frames[symbol][tf] = frame.loc[frame.index + pd.Timedelta(seconds=SECONDS[tf]) <= cutoff]
        name = manifest["funding"][symbol].get("filename", f"{symbol}_funding.csv")
        table = pd.read_csv(cache/name, index_col=0)
        table.index = pd.to_datetime(table.index, utc=True)
        funding[symbol] = table.iloc[:, 0].loc[table.index <= cutoff]
    return frames, funding


def register_plan(cache, out_dir, *, name, markets, modes, resolution, folds, contract_snapshot=None):
    """No strategy or PnL is evaluated during preregistration."""
    cache, out_dir = Path(cache).resolve(), Path(out_dir)
    target = out_dir/"research_plan.json"
    if target.exists():
        raise ValueError("Plan already registered; use another experiment directory")
    if resolution not in {"1m", "5m"} or not modes or len(set(modes)) != len(modes) or not set(modes) <= {"intraday", "scalp"}:
        raise ValueError("Invalid research mode or execution resolution")
    if resolution == "5m" and "scalp" in modes:
        raise ValueError("Scalp M1 confirmation requires M1 execution data")
    if not markets or len({market["name"] for market in markets}) != len(markets):
        raise ValueError("Markets must be unique and preregistered")
    for market in markets:
        if not market["symbols"] or len(set(market["symbols"])) != len(market["symbols"]):
            raise ValueError("Market symbol order must be nonempty and unique")
        for field in ("baseline_costs_bps_per_side", "stress_costs_bps_per_side"):
            costs = market.get(field, POLICY[field])
            if len(costs) != 2 or any(not math.isfinite(float(value)) or float(value) < 0 for value in costs):
                raise ValueError("Market costs must be two finite nonnegative bps amounts")
            market[field] = list(costs)
    folds = validate_folds(folds, resolution)
    manifest = json.loads((cache/"snapshot.json").read_text(encoding="utf-8"))
    if manifest.get("status", "complete") != "complete":
        raise ValueError("Only a complete verified dataset can be registered")
    venue = manifest.get("venue", "mexc")
    if venue != "mexc" and contract_snapshot is not None:
        raise ValueError("MEXC contract constraints cannot be applied to another venue")
    if utc(folds["test"]["end_utc"]) > utc(manifest["server_snapshot_utc"]):
        raise ValueError("Fold extends beyond the frozen source cutoff")
    symbols = list(dict.fromkeys(symbol for market in markets for symbol in market["symbols"]))
    paths = source_paths(cache, manifest, symbols, modes, resolution)
    hashes = {}
    for filename, (path, expected) in paths.items():
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"Modified source CSV: {filename}")
        hashes[filename] = actual
    root = Path(__file__).parent
    contracts = {}
    constraints_source = None
    if contract_snapshot is not None:
        artifact = json.loads(Path(contract_snapshot).read_text(encoding="utf-8"))
        records = {item["symbol"]: item for item in artifact["records"]}
        keys = ("contractSize", "volUnit", "minVol", "maxVol", "limitMaxVol", "priceUnit", "volScale", "priceScale")
        for symbol in symbols:
            if symbol not in records:
                raise ValueError(f"Contract constraint snapshot missing {symbol}")
            contracts[symbol] = {key: records[symbol][key] for key in keys if key in records[symbol]}
        constraints_source = {"sha256": file_hash(contract_snapshot), "observed_at_utc": artifact["observed_at_utc"],
                              "historical_constraints_known": False, "current_snapshot_proxy": True}
    adapter = "research_data.py" if venue == "mexc" else "external_data.py"
    plan = {"version": 1, "experiment": name, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "cache_path": str(cache), "source_venue": venue, "source_description": manifest.get("source"),
            "cross_venue_experiment": venue != "mexc", "never_claimed_as_mexc_pnl": venue != "mexc",
            "execution_resolution": resolution, "coarse_execution_proxy": resolution != "1m",
            "markets": markets, "market_trials": len(markets), "modes": modes, "folds": folds,
            "candidates": candidate_grid(), "policy": POLICY, "data_sha256": hashes,
            "data_manifest_sha256": file_hash(cache/"snapshot.json"),
            "source_snapshot_utc": manifest["server_snapshot_utc"],
            "code_sha256": {filename: file_hash(root/filename) for filename in CODE_FILES},
            "code_hash_scope": "executed strategy, account model, risk/config, and contract-rounding helpers",
            "data_adapter_provenance_sha256": {adapter: file_hash(root/adapter)},
            "position_limits": contracts, "contract_constraints_source": constraints_source,
            "settings_by_mode": {mode: settings_for(mode).model_dump(mode="json") for mode in modes},
            "previously_inspected": ["255 v1 experiments; selected-current universe", "BTC/ETH intraday 180d and scalp 30d",
                                    "365d swing aggregates/context", "30d shared-account M1 portfolios"],
            "holdout_is_not_guaranteed_pristine": True, "search_budget": {
                "train_main": 24*len(markets)*len(modes), "train_ablations": 6*len(markets)*len(modes),
                "train_controls": len(markets)*len(modes), "validation_max": 4*len(markets)*len(modes),
                "final_baseline_and_stress_max": 4*len(markets)*len(modes)}}
    # Coverage checks inspect only timestamps, not returns or strategy outcomes.
    frames, _ = read_dataset(plan)
    first, last = utc(folds["train"]["start_utc"]), utc(folds["test"]["end_utc"])
    index = pd.date_range(first, last-pd.Timedelta(seconds=SECONDS[resolution]), freq=f"{SECONDS[resolution]}s")
    plan["coverage"] = {}
    for symbol in symbols:
        frame = frames[symbol][resolution]
        missing = int((frame.index.get_indexer(index) < 0).sum())
        if missing:
            raise ValueError(f"Missing {missing} observed {resolution} execution bars: {symbol}")
        plan["coverage"][symbol] = {"execution_bars": len(index), "missing_execution_bars": missing,
                                   "warmup_closed_bars_by_tf": {tf: int((value.index+pd.Timedelta(seconds=SECONDS[tf]) <= first).sum())
                                                               for tf, value in frames[symbol].items()}}
    plan["plan_sha256"] = digest(plan)
    atomic_json(target, plan)
    protocol = root.parent/"docs"/"research_protocol.md"
    (out_dir/"registered_protocol.md").write_bytes(protocol.read_bytes())
    return plan


def verify_plan(plan):
    if digest({key: value for key, value in plan.items() if key != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("Registered plan was modified")
    root = Path(__file__).parent
    for filename, expected in plan["code_sha256"].items():
        if file_hash(root/filename) != expected:
            raise ValueError(f"Research source changed after registration: {filename}")


def wilson(successes, total, z=1.959963984540054):
    if not total:
        return None
    p, zz = successes/total, z*z
    center = (p+zz/(2*total))/(1+zz/total)
    half = z*math.sqrt(p*(1-p)/total+zz/(4*total*total))/(1+zz/total)
    return [max(0, center-half)*100, min(1, center+half)*100]


def annotate(result):
    trades = result["trades"]
    tp = sum(trade["status"] == "TP1_HIT" for trade in trades)
    n = len(trades)
    result.update({"tp_hits": tp, "tp_rate_percent": tp/n*100 if n else None,
                   "tp_wilson_95_percent": wilson(tp, n), "net_win_wilson_95_percent": wilson(result["wins"], n),
                   "long_trades": sum(trade["direction"] == "LONG" for trade in trades),
                   "short_trades": sum(trade["direction"] == "SHORT" for trade in trades),
                   "small_sample": n < 30})
    return result


def rank_value(result):
    pf = result["profit_factor"]
    if pf is None:
        pf = math.inf if result["trades_count"] and result["wins"] == result["trades_count"] else 0
    return pf, result["return_percent"]


def shortlist(results, policy=POLICY):
    eligible = [r for r in results if r["candidate"]["category"] == "candidate"
                and r["net_pnl_usdt"] > 0 and r["trades_count"] >= policy["train_min_trades"]
                and not r.get("unfinished_positions", 0)]
    return sorted(eligible, key=lambda r: (-rank_value(r)[0], -rank_value(r)[1], r["candidate"]["id"]))[:policy["shortlist_limit"]]


def validation_winner(results, training, policy=POLICY):
    train = {r["candidate"]["id"]: r for r in training}
    allowed = {r["candidate"]["id"] for r in shortlist(training, policy)}
    eligible = [r for r in results if r["candidate"]["id"] in allowed and r["net_pnl_usdt"] > 0
                and r["trades_count"] >= policy["validation_min_trades"] and not r.get("unfinished_positions", 0)]
    ordered = sorted(eligible, key=lambda r: (-rank_value(r)[0], -rank_value(r)[1],
                                             -rank_value(train[r["candidate"]["id"]])[0], r["candidate"]["id"]))
    return ordered[0] if ordered else None


def precompute_signals(frames, symbols, settings, candidates, window):
    """All threshold variants share one batch structural pass per closed bar."""
    from app.strategy.volium_v2 import V2Parameters, analyze_volium_v2_batch_from_df
    v2 = [candidate for candidate in candidates if candidate["parameters"] is not None]
    parameter_sets = [V2Parameters(**candidate["parameters"]) for candidate in v2]
    tables = {candidate["id"]: {} for candidate in candidates}
    tfs = required_timeframes(settings)
    entry_tf = tfs[-1]
    start, end = utc(window["start_utc"]), utc(window["end_utc"])
    total_checks, batch_calls = 0, 0
    for symbol in symbols:
        closes = {tf: frames[symbol][tf].index+pd.Timedelta(seconds=SECONDS[tf]) for tf in tfs}
        times = closes[entry_tf][(closes[entry_tf] > start) & (closes[entry_tf] <= end)]
        for index, now in enumerate(times, 1):
            if index % 10000 == 0:
                print(f"signals {settings.volium_mode} {symbol}: {index}/{len(times)} closed bars", flush=True)
            if settings.volium_session_enabled and not in_volium_session(now, settings):
                continue
            known = {tf: frames[symbol][tf].iloc[:closes[tf].searchsorted(now, side="right")].tail(settings.volium_context_lookback+30)
                     for tf in tfs}
            kwargs = {"symbol": symbol, "frames": known, "settings": settings,
                      "mode": settings.volium_mode, "now": now.to_pydatetime()}
            if parameter_sets:
                signals = analyze_volium_v2_batch_from_df(parameter_sets=parameter_sets, **kwargs)
                if len(signals) != len(v2):
                    raise ValueError("V2 batch result length differs from preregistered candidates")
                batch_calls += 1
                for candidate, signal in zip(v2, signals):
                    if signal is not None:
                        tables[candidate["id"]][(symbol, now.value)] = signal.model_dump(mode="json")
            if "v1-control" in tables:
                signal = analyze_volium_from_df(**kwargs)
                if signal is not None:
                    tables["v1-control"][(symbol, now.value)] = signal.model_dump(mode="json")
            total_checks += 1
    return tables, {"eligible_time_boundaries": total_checks, "batch_structural_calls": batch_calls,
                    "positive_outputs_by_candidate": {key: len(value) for key, value in tables.items()}}


def run_group(plan, phase, market, mode, candidates, checkpoint):
    verify_plan(plan)
    window = plan["folds"]["test" if phase == "final" else phase]
    frames, funding = read_dataset(plan, end_at=window["end_utc"])
    symbols = market["symbols"]
    settings = Settings(_env_file=None, **plan["settings_by_mode"][mode])
    tables, metrics = precompute_signals(frames, symbols, settings, candidates, window)
    payload = {"market": market["name"], "mode": mode, "phase": phase, "plan_sha256": plan["plan_sha256"],
               "signal_cache": metrics, "results": []}
    for candidate in candidates:
        signals = tables[candidate["id"]]
        def provider(**kwargs):
            value = signals.get((kwargs["symbol"], utc(kwargs["now"]).value))
            return TradeSetup.model_validate(value) if value is not None else None
        for stress in ((False, True) if phase == "final" else (False,)):
            started = time.monotonic()
            costs = market["stress_costs_bps_per_side" if stress else "baseline_costs_bps_per_side"]
            cost_settings = settings.model_copy(update={"paper_fee_bps": costs[0], "paper_slippage_bps": costs[1]})
            result = portfolio_replay({symbol: frames[symbol] for symbol in symbols}, cost_settings,
                symbols=symbols, start_at=window["start_utc"], end_at=window["end_utc"],
                execution_frames={symbol: frames[symbol][plan["execution_resolution"]] for symbol in symbols},
                funding_rates={symbol: funding[symbol] for symbol in symbols}, signal_provider=provider,
                execution_timeframe=plan["execution_resolution"],
                position_limits={symbol: plan.get("position_limits", {})[symbol] for symbol in symbols
                                 if symbol in plan.get("position_limits", {})},
                signal_schedule={symbol: [pd.Timestamp(nanos, tz="UTC") for item_symbol, nanos in signals
                                          if item_symbol == symbol] for symbol in symbols})
            result.update({"candidate": candidate, "phase": phase, "market": market["name"],
                           "source_venue": plan["source_venue"], "cost_variant": "stress" if stress else "baseline",
                           "requested_window": window, "elapsed_seconds": round(time.monotonic()-started, 3)})
            payload["results"].append(annotate(result))
            atomic_json(checkpoint, payload)
            print(f"{phase} {market['name']} {mode} {candidate['id']} {result['cost_variant']}: "
                  f"n={result['trades_count']} TP={result['tp_hits']} net={result['net_pnl_usdt']:.2f}", flush=True)
    return payload


def group_results(results, market, mode):
    return [result for result in results if result["market"] == market and result["mode"] == mode]


def result_identity(result):
    return (result["market"], result["mode"], result["candidate"]["id"], result["cost_variant"])


def final_jobs(plan, selection):
    jobs = []
    control = next(candidate for candidate in plan["candidates"] if candidate["id"] == "v1-control")
    for market in plan["markets"]:
        for mode in plan["modes"]:
            chosen = selection["winners"][market["name"]+"/"+mode]
            if chosen is not None:
                jobs.append((market, mode, [chosen, control]))
    return jobs


def freeze_selection(plan, out_dir, train, validation):
    winners = {}
    for market in plan["markets"]:
        for mode in plan["modes"]:
            winner = validation_winner(group_results(validation, market["name"], mode),
                                       group_results(train, market["name"], mode), plan["policy"])
            winners[market["name"]+"/"+mode] = winner["candidate"] if winner else None
    selection = {"plan_sha256": plan["plan_sha256"], "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                 "train_results_sha256": file_hash(Path(out_dir)/"train_results.json"),
                 "validation_results_sha256": file_hash(Path(out_dir)/"validation_results.json"),
                 "winners": winners, "test_outcomes_opened": False}
    selection["selection_sha256"] = digest(selection)
    target = Path(out_dir)/"frozen_selection.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        validate_selection(plan, out_dir, existing)
        if existing["winners"] != winners or existing["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("Selection already frozen; final outcomes cannot change it")
        return existing
    if (Path(out_dir)/"final_opened.json").exists():
        raise ValueError("Cannot recreate missing selection after final outcomes opened")
    atomic_json(target, selection)
    return selection


def validate_selection(plan, out_dir, selection):
    if selection["plan_sha256"] != plan["plan_sha256"] or digest({k: v for k, v in selection.items()
            if k != "selection_sha256"}) != selection["selection_sha256"]:
        raise ValueError("Frozen validation selection was modified")
    for stage in ("train", "validation"):
        if file_hash(Path(out_dir)/f"{stage}_results.json") != selection[f"{stage}_results_sha256"]:
            raise ValueError("Selection inputs changed after freeze")
    marker = Path(out_dir)/"final_opened.json"
    expected = {"plan_sha256": plan["plan_sha256"], "selection_sha256": selection["selection_sha256"]}
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != expected:
        raise ValueError("Final outcomes already opened under a different selection")


def load_stage(out_dir, phase, plan):
    value = json.loads((Path(out_dir)/f"{phase}_results.json").read_text(encoding="utf-8"))
    if value["plan_sha256"] != plan["plan_sha256"] or not value.get("completed_at_utc"):
        raise ValueError(f"Incomplete or mismatching {phase} stage")
    return value


def research_verdicts(plan, selection, results):
    verdicts = {}
    for group, candidate in selection["winners"].items():
        market, mode = group.split("/", 1)
        if candidate is None:
            verdicts[group] = {"status": "no_validated_candidate", "positive_historical_candidate": False,
                               "final_test_opened_for_group": False}
            continue
        rows = {(row["phase"], row["cost_variant"]): row for row in results
                if row["market"] == market and row["mode"] == mode and row["candidate"]["id"] == candidate["id"]}
        baseline = [rows.get((phase, "baseline")) for phase in ("train", "validation", "final")]
        if any(row is None for row in baseline):
            verdicts[group] = {"status": "frozen_pending_final", "candidate": candidate}
            continue
        floors = [plan["policy"][key] for key in ("train_min_trades", "validation_min_trades", "test_min_trades")]
        positive = all(row["net_pnl_usdt"] > 0 and row["trades_count"] >= floor
                       and not row.get("unfinished_positions", 0) for row, floor in zip(baseline, floors))
        stress = rows.get(("final", "stress"))
        survived = bool(stress is not None and stress["net_pnl_usdt"] >= 0 and not stress.get("unfinished_positions", 0))
        small = any(row["trades_count"] < plan["policy"]["small_sample_below"] for row in baseline)
        verdicts[group] = {"status": "positive_historical_candidate_with_small_sample" if positive and small
                          else "positive_historical_candidate" if positive else "failed_final_criteria",
                          "candidate": candidate, "positive_historical_candidate": positive,
                          "cost_stress_survived": survived, "small_sample": small,
                          "fold_trades": [row["trades_count"] for row in baseline],
                          "fold_net_pnl_usdt": [row["net_pnl_usdt"] for row in baseline]}
    return verdicts


def write_report(out_dir, plan):
    results, stages = [], {}
    for phase in ("train", "validation", "final"):
        path = Path(out_dir)/f"{phase}_results.json"
        if path.exists():
            stage = json.loads(path.read_text(encoding="utf-8"))
            stages[phase] = {"completed": bool(stage.get("completed_at_utc")), "trials": len(stage["results"])}
            results.extend(stage["results"])
    summary = []
    for r in results:
        summary.append({"phase": r["phase"], "venue": r["source_venue"], "market": r["market"], "mode": r["mode"],
                        "candidate": r["candidate"]["id"], "category": r["candidate"]["category"],
                        "costs": r["cost_variant"], "execution_resolution": r["execution_timeframe"],
                        "trades": r["trades_count"], "long": r["long_trades"], "short": r["short_trades"],
                        "tp": r["tp_hits"], "tp_rate": r["tp_rate_percent"], "net_win_rate": r["win_rate"],
                        "net_pnl_usdt": r["net_pnl_usdt"], "return_percent": r["return_percent"],
                        "funding_usdt": r["funding_total_usdt"], "profit_factor": r["profit_factor"],
                        "marked_drawdown_percent": r["max_marked_drawdown_percent"], "marked_equity": r["marked_equity"],
                        "unfilled_signals": r["unfilled_signals"], "small_sample": r["small_sample"]})
    pd.DataFrame(summary).to_csv(Path(out_dir)/"research_summary.csv", index=False, encoding="utf-8-sig")
    lines = [f"# Исследование {plan['experiment']}", "",
             f"Площадка: **{plan['source_venue']}**; исполнение: **{plan['execution_resolution']}**; "
             f"рыночных гипотез: {plan['market_trials']}. План SHA256: {plan['plan_sha256']}.", ""]
    if plan["cross_venue_experiment"]:
        lines += ["Это отдельный эксперимент другой площадки, не доходность MEXC. Funding взят с той же площадки, что цены.", ""]
    if plan["coarse_execution_proxy"]:
        lines += ["Исполнение M5 — грубая OHLC-модель. Минутные пути внутри свечей не выдумываются.", ""]
    for name, fold in plan["folds"].items():
        lines.append(f"{name}: {fold['start_utc']} → {fold['end_utc']}, {fold['execution_bars']} свечей исполнения.")
    for market in plan["markets"]:
        lines.append(f"{market['name']}: {', '.join(market['symbols'])}; fee+slip bps/side "
                     f"{market['baseline_costs_bps_per_side']} baseline / {market['stress_costs_bps_per_side']} stress.")
    if plan.get("position_limits"):
        lines += ["Округление цен и объёма и пределы контрактов взяты из текущего снимка; исторические ограничения неизвестны. "
                  "Издержки общего FX-портфеля — консервативная гипотеза для всех его инструментов; эффект нельзя приписать только FX."]
    lines += ["", "Фактически выполненные испытания: " + canonical(stages) + ".",
              "24 основных варианта, 6 семантических ablation и v1 зарегистрированы до поиска. "
              "Train-shortlist: net>0 и n≥10; validation/test: net>0 и n≥3. Это разведочные минимумы. "
              "Менее 30 сделок в этапе означает малую выборку; положительный исторический кандидат не является доказанным преимуществом.", "",
              "| Этап | Рынок/режим | Кандидат | Издержки | Сделки L/S | TP | Net USDT | DD% |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['phase']} | {r['market']}/{r['mode']} | {r['candidate']['id']} | {r['cost_variant']} | "
                     f"{r['trades_count']} ({r['long_trades']}/{r['short_trades']}) | {r['tp_hits']} | "
                     f"{r['net_pnl_usdt']:.2f} | {r['max_marked_drawdown_percent']:.2f} |")
    selection_path = Path(out_dir)/"frozen_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        verdicts = research_verdicts(plan, selection, results)
        atomic_json(Path(out_dir)/"research_verdicts.json", {"plan_sha256": plan["plan_sha256"], "groups": verdicts})
        lines += ["", "Замороженный выбор: " + canonical(selection["winners"]) + "."]
        if not any(selection["winners"].values()):
            lines += ["Ни один кандидат не прошёл validation; финальный участок не использован для подгонки."]
        for group, verdict in verdicts.items():
            lines.append(f"{group}: **{verdict['status']}**; " + canonical(verdict) + ".")
    lines += ["", "Все полные сделки, funding, TP/net-win Wilson-интервалы и отмеченные открытые позиции находятся в JSON. "
              "История и текущий состав рынка уже частично изучались; holdout не гарантированно неизвестен. "
              "Конечный бюджет попыток и разделение истории не гарантируют отсутствие переобучения. "
              "Просадка наблюдается на закрытиях OHLC; очередь лимитного исполнения и ликвидация не моделируются.", ""]
    (Path(out_dir)/"research_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_phase(out_dir, phase, workers=1):
    out_dir = Path(out_dir)
    plan = json.loads((out_dir/"research_plan.json").read_text(encoding="utf-8"))
    verify_plan(plan)
    output = out_dir/f"{phase}_results.json"
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved.get("completed_at_utc"):
            if saved["plan_sha256"] != plan["plan_sha256"]:
                raise ValueError("Completed stage belongs to another plan")
            if phase == "validation":
                training = load_stage(out_dir, "train", plan)["results"]
                freeze_selection(plan, out_dir, training, saved["results"])
            elif phase == "final":
                validate_selection(plan, out_dir, json.loads((out_dir/"frozen_selection.json").read_text(encoding="utf-8")))
            write_report(out_dir, plan)
            print(f"{phase} already completed; returning immutable recorded results", flush=True)
            return saved
    if phase == "train":
        if (out_dir/"frozen_selection.json").exists():
            raise ValueError("Train cannot be reopened after validation selection")
        jobs = [(market, mode, plan["candidates"]) for market in plan["markets"] for mode in plan["modes"]]
    elif phase == "validation":
        training = load_stage(out_dir, "train", plan)["results"]
        control = next(c for c in plan["candidates"] if c["id"] == "v1-control")
        jobs = [(market, mode, [r["candidate"] for r in shortlist(group_results(training, market["name"], mode), plan["policy"])]+[control])
                for market in plan["markets"] for mode in plan["modes"]]
    elif phase == "final":
        selection = json.loads((out_dir/"frozen_selection.json").read_text(encoding="utf-8"))
        validate_selection(plan, out_dir, selection)
        jobs = final_jobs(plan, selection)
        # Durable opening marker precedes any final-test computation.
        marker = {"plan_sha256": plan["plan_sha256"], "selection_sha256": selection["selection_sha256"]}
        marker_path = out_dir/"final_opened.json"
        if marker_path.exists() and json.loads(marker_path.read_text(encoding="utf-8")) != marker:
            raise ValueError("Final outcomes already opened under a different selection")
        if jobs:
            atomic_json(marker_path, marker)
    else:
        raise ValueError("Unknown research stage")
    expected = {(market["name"], mode, candidate["id"], cost) for market, mode, candidates in jobs
                for candidate in candidates for cost in (("baseline", "stress") if phase == "final" else ("baseline",))}
    payload = {"phase": phase, "plan_sha256": plan["plan_sha256"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "expected_trials": len(expected), "results": [], "signal_cache": {}}
    checkpoints = out_dir/"work"/phase
    checkpoints.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs) or 1))) as pool:
        pending = {pool.submit(run_group, plan, phase, market, mode, candidates,
                               str(checkpoints/f"{market['name']}_{mode}.json"))
                   for market, mode, candidates in jobs}
        authoritative = []
        previous_count = -1
        while pending:
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            authoritative.extend(future.result() for future in done)
            results = []
            for market, mode, _ in jobs:
                path = checkpoints/f"{market['name']}_{mode}.json"
                if path.exists():
                    checkpoint = json.loads(path.read_text(encoding="utf-8"))
                    if checkpoint["plan_sha256"] != plan["plan_sha256"]:
                        raise ValueError("Checkpoint belongs to another plan")
                    results.extend(checkpoint["results"])
            if len(results) != previous_count:
                payload["results"] = results
                atomic_json(output, payload)
                write_report(out_dir, plan)
                print(f"{phase} checkpoint {len(results)}/{len(expected)}", flush=True)
                previous_count = len(results)
            if not pending:
                by_case = {}
                for value in authoritative:
                    payload["signal_cache"][value["market"]+"/"+value["mode"]] = value["signal_cache"]
                    for result in value["results"]:
                        key = result_identity(result)
                        if key in by_case:
                            raise ValueError("Duplicate completed research trial")
                        by_case[key] = result
                if set(by_case) != expected:
                    raise ValueError("Completed research trials do not match the preregistered stage")
                payload["results"] = list(by_case.values())
                break
    if len(payload["results"]) != len(expected):
        raise ValueError("Research stage incomplete")
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output, payload)
    if phase == "validation":
        freeze_selection(plan, out_dir, training, payload["results"])
    write_report(out_dir, plan)
    print(f"Completed {phase}: {len(expected)} preregistered trials", flush=True)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("draft", "register", "train", "validation", "final"))
    parser.add_argument("--cache")
    parser.add_argument("--out-dir", default="outputs/research")
    parser.add_argument("--name", default="volium-v2-research")
    parser.add_argument("--market", action="append", help="name=SYMBOL1,SYMBOL2; preregister group and order")
    parser.add_argument("--market-costs", action="append", help="name=baselineFee,baselineSlip,stressFee,stressSlip in bps/side")
    parser.add_argument("--contract-snapshot", help="Current public contract constraint JSON; historical proxy, MEXC only")
    parser.add_argument("--mode", choices=("intraday", "scalp"), action="append")
    parser.add_argument("--execution-resolution", choices=("1m", "5m"), default="1m")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--fold", action="append", help="train|validation|test=START/END; explicit contiguous folds")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args(argv)
    if args.phase == "draft":
        print(json.dumps({"candidates": candidate_grid(), "policy": POLICY,
                          "stage_order": ["register", "train", "validation", "final"]}, indent=2))
        return
    if args.phase == "register":
        if not args.cache or not args.market:
            parser.error("register requires --cache and predeclared --market")
        markets = [{"name": item.split("=", 1)[0], "symbols": item.split("=", 1)[1].split(",")}
                   for item in args.market]
        for item in args.market_costs or []:
            market_name, numbers = item.split("=", 1)
            values = [float(number) for number in numbers.split(",")]
            if len(values) != 4 or market_name not in {market["name"] for market in markets}:
                parser.error("--market-costs requires a declared market and four bps values")
            for market in markets:
                if market["name"] == market_name:
                    market["baseline_costs_bps_per_side"] = values[:2]
                    market["stress_costs_bps_per_side"] = values[2:]
        if args.fold:
            folds = {}
            for item in args.fold:
                stage, bounds = item.split("=", 1)
                start, end = bounds.split("/", 1)
                folds[stage] = {"start_utc": start, "end_utc": end}
        elif args.start and args.end:
            folds = chronological_folds(args.start, args.end, args.execution_resolution)
        else:
            parser.error("register requires --start/--end or three explicit --fold boundaries")
        plan = register_plan(args.cache, args.out_dir, name=args.name, markets=markets,
                             modes=args.mode or ["intraday"], resolution=args.execution_resolution, folds=folds,
                             contract_snapshot=args.contract_snapshot)
        print(f"Registered {plan['plan_sha256']} without inspecting candidate outcomes")
    else:
        run_phase(args.out_dir, args.phase, args.workers)


if __name__ == "__main__":
    main()
