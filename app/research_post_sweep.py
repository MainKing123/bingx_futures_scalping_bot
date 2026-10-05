"""Final bounded post-sweep-break context study; prior engines stay intact.

No search runs implicitly. Register a new source-reviewed plan, then run its
train/validation/final stages. Providers are named explicitly in each trial.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app import research_suite as common
from app.research_followup import ProviderSpec, provider_functions, parent_trial_ledger, enforce_parent_design, stage_jobs
from app.config import Settings
from app.portfolio_replay import portfolio_replay
from app.replay import SECONDS, required_timeframes
from app.schemas.setup import TradeSetup
from app.strategy.volium import in_volium_session


PROVIDERS = {
    "v4": ProviderSpec("experimental_v4", "app.strategy.volium_v4", "V4Parameters",
                       "analyze_volium_v4_batch_from_df", "diagnose_volium_v4_batch_from_df", h1_history_daily_multiplier=24),
    "v3_daily": ProviderSpec("fixed_v3_daily_control", "app.strategy.volium_v3", "V3Parameters",
                             "analyze_volium_v3_batch_from_df", "diagnose_volium_v3_batch_from_df", h1_history_daily_multiplier=24),
    "v3_local": ProviderSpec("fixed_v3_local_control", "app.strategy.volium_v3", "V3Parameters",
                             "analyze_volium_v3_batch_from_df", "diagnose_volium_v3_batch_from_df", h1_history_daily_multiplier=24),
    "v1": ProviderSpec("original_v1", "app.strategy.volium", None, None, None, "analyze_volium_from_df"),
}
CONTEXTS = ("post_sweep_break",)
JOINT_POST_SWEEP_BUDGET = {"predeclared_venues": ["mexc", "binance_usdm"],
                         "mexc_market_mode_groups": 2, "binance_intraday_market_mode_groups": 1,
                         "train_main": 72, "train_controls": 9, "train_total": 81,
                         "validation_max": 18, "final_baseline_and_stress_max": 24,
                         "last_context_family_for_this_research": True}
DEFAULTS = {"context_mode": "post_sweep_break", "correction_target_mode": "leg_origin",
            "reaction_atr_period": 14, "reaction_min_atr": .8, "reaction_min_body_ratio": .6,
            "reaction_min_prior_body_ratio": 1., "reaction_max_bars": 3}


def post_sweep_grid():
    """One predeclared context, unchanged 24 strength cells, three fixed controls."""
    candidates = []
    for atr, body, prior, bars in itertools.product((.8, 1.2, 1.6), (.6, .7), (1., 1.5), (2, 3)):
        parameters = {**DEFAULTS, "reaction_min_atr": atr, "reaction_min_body_ratio": body,
                      "reaction_min_prior_body_ratio": prior, "reaction_max_bars": bars}
        identity = {"provider": asdict(PROVIDERS["v4"]), "parameters": parameters}
        candidates.append({"id": "v4-"+common.digest(identity)[:16], "category": "candidate",
                           "provider": "v4", "parameters": parameters})
    candidates.append({"id": "v1-control", "category": "control", "provider": "v1", "parameters": None})
    for provider, context in (("v3_daily", "latest_leg_daily"), ("v3_local", "latest_leg_local")):
        candidates.append({"id": provider.replace("_", "-")+"-control", "category": "control",
                           "provider": provider, "parameters": {**DEFAULTS, "context_mode": context}})
    return candidates


def verify_post_sweep_plan(plan):
    common.verify_plan(plan)
    for parent in plan["parent_trials"]:
        root = Path(parent["directory"])
        if common.file_hash(root/"research_plan.json") != parent["plan_file_sha256"]:
            raise ValueError("Recorded parent plan changed after post-sweep registration")
        for phase, metadata in parent["phases"].items():
            if common.file_hash(root/f"{phase}_results.json") != metadata["sha256"]:
                raise ValueError("Recorded parent research outcomes changed")
        if (root/"final_opened.json").exists() != parent["final_test_opened"]:
            raise ValueError("Recorded parent final-test inspection changed")


def completed_parent_ledger(directories):
    """Both V2 and V3 venues must be completed before this final family."""
    ledger = parent_trial_ledger(directories)
    profiles = set()
    for parent in ledger:
        plan = json.loads((Path(parent["directory"])/"research_plan.json").read_text(encoding="utf-8"))
        runner = plan.get("runner", "app.research_suite")
        if runner not in {"app.research_suite", "app.research_followup"}:
            raise ValueError("Only recorded V2/V3 parents belong to this protocol")
        profiles.add((runner, parent["source_venue"]))
        if set(parent["phases"]) != {"train", "validation", "final"}:
            raise ValueError("Complete every parent stage, including zero-trial final")
    expected = {(runner, venue) for runner in ("app.research_suite", "app.research_followup")
                for venue in ("mexc", "binance_usdm")}
    if len(ledger) != 4 or profiles != expected:
        raise ValueError("All four completed V2/V3 venue plans are required")
    return ledger


def archive_provenance(archive, engine_files):
    archive = Path(archive)
    manifest_path = archive.with_name(archive.stem+"_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if common.file_hash(archive) != manifest["sha256"] or archive.stat().st_size != manifest["bytes"]:
        raise ValueError("Prior source archive differs from its manifest")
    records = {record["path"]: record for record in manifest["files"]}
    if len(records) != len(manifest["files"]):
        raise ValueError("Duplicate archived source path")
    with zipfile.ZipFile(archive) as bundle:
        if set(bundle.namelist()) != set(records) or len(bundle.namelist()) != len(records):
            raise ValueError("Archive entries differ from per-file manifest")
        for name, metadata in records.items():
            data = bundle.read(name)
            if len(data) != metadata["bytes"] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
                raise ValueError("Archived source bytes differ: "+name)
    source_root = Path(__file__).parent
    for filename in engine_files:
        if filename in {"research_post_sweep.py", "strategy/volium_v4.py"}:
            continue
        record = records.get("mexc-volium/app/"+filename)
        if record is None or common.file_hash(source_root/filename) != record["sha256"]:
            raise ValueError("Frozen prior engine differs from archived source: "+filename)
    return {"filename": archive.name, "sha256": manifest["sha256"],
            "manifest_sha256": common.file_hash(manifest_path), "verified_file_count": len(records)}


def register_post_sweep(cache, out_dir, *, name, markets, modes, resolution, folds,
                      parent_directories, source_review, source_archive, peer_out_dir, contract_snapshot=None):
    """Only source API/data/timestamp checks run here; no candidate outcomes."""
    cache, out_dir = Path(cache).resolve(), Path(out_dir)
    if (out_dir/"research_plan.json").exists():
        raise ValueError("Follow-up plan already registered; never overwrite prior research")
    if Path(peer_out_dir).resolve() == out_dir.resolve():
        raise ValueError("The paired venue plan must use a separate directory")
    if not parent_directories or not Path(source_review).is_file() or not Path(source_archive).is_file():
        raise ValueError("Follow-up requires prior-trial ledger, reviewed source evidence and preserved old source archive")
    if resolution not in {"1m", "5m"} or modes != ["intraday"]:
        raise ValueError("Invalid post-sweep modes/resolution")
    if not markets or len({market["name"] for market in markets}) != len(markets):
        raise ValueError("Market hypotheses must be finite and unique")
    for market in markets:
        if not market["symbols"] or len(set(market["symbols"])) != len(market["symbols"]):
            raise ValueError("Market symbols must be unique and ordered")
        for key in ("baseline_costs_bps_per_side", "stress_costs_bps_per_side"):
            costs = market.get(key, common.POLICY[key])
            if len(costs) != 2 or any(not math.isfinite(float(value)) or float(value) < 0 for value in costs):
                raise ValueError("Costs must be finite nonnegative fee/slip pairs")
            market[key] = list(costs)
    folds = common.validate_folds(folds, resolution)
    manifest = json.loads((cache/"snapshot.json").read_text(encoding="utf-8"))
    if manifest.get("status", "complete") != "complete":
        raise ValueError("Verified complete data required")
    venue = manifest.get("venue", "mexc")
    if venue != "mexc" and contract_snapshot is not None:
        raise ValueError("MEXC contract rules cannot be mixed with another venue")
    if common.utc(folds["test"]["end_utc"]) > common.utc(manifest["server_snapshot_utc"]):
        raise ValueError("Outcomes extend beyond the frozen source cutoff")
    grid = post_sweep_grid()
    specs = {key: asdict(value) for key, value in PROVIDERS.items()}
    root = Path(__file__).parent
    engine_files = tuple(dict.fromkeys((*common.CODE_FILES, "research_followup.py", "research_post_sweep.py",
                                       *(spec.module.removeprefix("app.").replace(".", "/")+".py" for spec in PROVIDERS.values()))))
    for candidate in grid:
        factory, _, _, _ = provider_functions(PROVIDERS[candidate["provider"]])
        if factory is not None:
            factory(**candidate["parameters"])
    symbols = list(dict.fromkeys(symbol for market in markets for symbol in market["symbols"]))
    hashes = {}
    for filename, (path, expected) in common.source_paths(cache, manifest, symbols, modes, resolution).items():
        if common.file_hash(path) != expected:
            raise ValueError("Modified source CSV: "+filename)
        hashes[filename] = expected
    contracts, contract_source = {}, None
    if contract_snapshot is not None:
        artifact = json.loads(Path(contract_snapshot).read_text(encoding="utf-8"))
        records = {item["symbol"]: item for item in artifact["records"]}
        keys = ("contractSize", "volUnit", "minVol", "maxVol", "limitMaxVol", "priceUnit", "volScale", "priceScale")
        for symbol in symbols:
            if symbol not in records:
                raise ValueError("Contract snapshot missing "+symbol)
            contracts[symbol] = {key: records[symbol][key] for key in keys if key in records[symbol]}
        contract_source = {"sha256": common.file_hash(contract_snapshot), "observed_at_utc": artifact["observed_at_utc"],
                           "historical_constraints_known": False, "current_snapshot_proxy": True}
    ledger = completed_parent_ledger(parent_directories)
    archive_metadata = archive_provenance(source_archive, engine_files)
    design_parent = enforce_parent_design(parent_directories, venue=venue,
        manifest_sha256=common.file_hash(cache/"snapshot.json"), markets=markets, modes=modes,
        resolution=resolution, folds=folds, contracts=contracts)
    adapter = "research_data.py" if venue == "mexc" else "external_data.py"
    strata = len(markets)*len(modes)
    plan = {"version": 4, "runner": "app.research_post_sweep", "experiment": name,
            "registered_at_utc": datetime.now(timezone.utc).isoformat(), "cache_path": str(cache),
            "source_venue": venue, "source_description": manifest.get("source"),
            "cross_venue_experiment": venue != "mexc", "never_claimed_as_mexc_pnl": venue != "mexc",
            "execution_resolution": resolution, "coarse_execution_proxy": resolution != "1m",
            "markets": markets, "market_trials": len(markets), "modes": modes, "folds": folds,
            "candidates": grid, "provider_specs": specs, "policy": dict(common.POLICY),
            "data_sha256": hashes, "data_manifest_sha256": common.file_hash(cache/"snapshot.json"),
            "source_snapshot_utc": manifest["server_snapshot_utc"],
            "code_sha256": {filename: common.file_hash(root/filename) for filename in engine_files},
            "code_hash_scope": "post-sweep runner, immutable shared helpers/engines, all explicit providers and contract helpers",
            "data_adapter_provenance_sha256": {adapter: common.file_hash(root/adapter)},
            "settings_by_mode": {mode: common.settings_for(mode).model_dump(mode="json") for mode in modes},
            "position_limits": contracts, "contract_constraints_source": contract_source,
            "source_review_sha256": common.file_hash(source_review), "prior_source_archive_sha256": common.file_hash(source_archive),
            "registered_protocol_sha256": common.file_hash(root.parent/"docs"/"research_post_sweep_protocol.md"),
            "prior_source_archive_provenance": archive_metadata,
            "parent_trials": ledger, "prior_v1_suite_trials": 255,
            "design_parent_plan_sha256": design_parent,
            "joint_post_sweep_budget": dict(JOINT_POST_SWEEP_BUDGET),
            "paired_plan_directory": str(Path(peer_out_dir).resolve()),
            "venue_selection_never_uses_other_venue_outcomes": True,
            "prior_recorded_research_trials": sum(phase["trials"] for parent in ledger for phase in parent["phases"].values()),
            "reuses_parent_train_and_validation": True, "holdout_is_not_guaranteed_pristine": True,
            "causal_history_bars_by_provider": {name:{"1d":110,"1h":2030 if PROVIDERS[name].h1_history_daily_multiplier else 110,"5m":110}
                                                for name in PROVIDERS},
            "local_control_requires_observed_h1_history": True,
            "strict_c_definition": "Immediate closed D1 after A: LONG HH/HL and close>A.high; SHORT mirrored",
            "post_a_break_may_be_countertrend": True, "not_full_author_replica": True,
            "supplemental_validation_source_inspection": "BTC 2025-06-24 author case: causal features and known author outcome, no replay/forward/final",
            "old_v1_shared_portfolios_disclosed_separately": 2,
            "prior_stage_trials_total": 255+sum(phase["trials"] for parent in ledger for phase in parent["phases"].values()),
            "search_budget": {"train_main": 24*strata, "train_controls": 3*strata,
                              "validation_max": 6*strata, "final_baseline_and_stress_max": 8*strata}}
    frames, _ = common.read_dataset(plan)
    first, last = common.utc(folds["train"]["start_utc"]), common.utc(folds["test"]["end_utc"])
    clock = pd.date_range(first, last-pd.Timedelta(seconds=SECONDS[resolution]), freq=f"{SECONDS[resolution]}s")
    plan["coverage"] = {}
    for symbol in symbols:
        if (frames[symbol][resolution].index.get_indexer(clock) < 0).any():
            raise ValueError("Incomplete observed execution data: "+symbol)
        plan["coverage"][symbol] = {"execution_bars": len(clock), "missing_execution_bars": 0,
            "warmup_closed_bars_by_tf": {tf: int((frame.index+pd.Timedelta(seconds=SECONDS[tf]) <= first).sum())
                                        for tf, frame in frames[symbol].items()}}
    plan["plan_sha256"] = common.digest(plan)
    common.atomic_json(out_dir/"research_plan.json", plan)
    (out_dir/"registered_protocol.md").write_bytes((root.parent/"docs"/"research_post_sweep_protocol.md").read_bytes())
    return plan


def precompute_post_sweep(frames, symbols, settings, candidates, window, specs):
    """Group ALL variants by provider, never reuse one variant's first signal."""
    tables = {candidate["id"]: {} for candidate in candidates}
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate["provider"], []).append(candidate)
    bundles = {}
    for name, group in groups.items():
        factory, batch, _, single = provider_functions(ProviderSpec(**specs[name]))
        parameters = [factory(**candidate["parameters"]) for candidate in group] if factory is not None else []
        bundles[name] = (parameters, batch, single)
    tfs = required_timeframes(settings)
    start, end = common.utc(window["start_utc"]), common.utc(window["end_utc"])
    calls, checks = {name: 0 for name in groups}, 0
    for symbol in symbols:
        closes = {tf: frames[symbol][tf].index+pd.Timedelta(seconds=SECONDS[tf]) for tf in tfs}
        times = closes[tfs[-1]][(closes[tfs[-1]] > start) & (closes[tfs[-1]] <= end)]
        for index, now in enumerate(times, 1):
            if index % 10000 == 0:
                print(f"post-sweep signals {settings.volium_mode} {symbol}: {index}/{len(times)}", flush=True)
            if settings.volium_session_enabled and not in_volium_session(now, settings):
                continue
            known = {tf: frames[symbol][tf].iloc[:closes[tf].searchsorted(now, side="right")].tail(
                         settings.volium_context_lookback*24+settings.volium_context_lookback+30
                         if tf == "1h" else settings.volium_context_lookback+30) for tf in tfs}
            kwargs = {"symbol": symbol, "frames": known, "settings": settings, "mode": settings.volium_mode,
                      "now": now.to_pydatetime()}
            for name, group in groups.items():
                parameters, batch, single = bundles[name]
                spec = ProviderSpec(**specs[name])
                group_known = {tf: frame.tail(settings.volium_context_lookback+30) if tf == "1h"
                               and not spec.h1_history_daily_multiplier else frame for tf, frame in known.items()}
                provider_kwargs = {**kwargs,"frames":group_known}
                values = batch(parameter_sets=parameters, **provider_kwargs) if batch is not None else [single(**provider_kwargs)]
                if len(values) != len(group):
                    raise ValueError("Provider batch length differs from the registered group")
                calls[name] += 1
                for candidate, signal in zip(group, values):
                    if signal is not None:
                        if signal.symbol != symbol or common.utc(signal.timestamp) != now:
                            raise ValueError("Provider signal must match current observed symbol/close")
                        tables[candidate["id"]][(symbol, now.value)] = signal.model_dump(mode="json")
            checks += 1
    return tables, {"eligible_time_boundaries": checks, "batch_calls_by_provider": calls,
                    "positive_outputs_by_candidate": {key: len(value) for key, value in tables.items()}}


def run_post_sweep_group(plan, phase, market, mode, candidates, checkpoint):
    verify_post_sweep_plan(plan)
    window = plan["folds"]["test" if phase == "final" else phase]
    frames, funding = common.read_dataset(plan, end_at=window["end_utc"])
    symbols = market["symbols"]
    settings = Settings(_env_file=None, **plan["settings_by_mode"][mode])
    tables, metrics = precompute_post_sweep(frames, symbols, settings, candidates, window, plan["provider_specs"])
    payload = {"market": market["name"], "mode": mode, "phase": phase, "plan_sha256": plan["plan_sha256"],
               "signal_cache": metrics, "results": []}
    for candidate in candidates:
        signals = tables[candidate["id"]]
        def provider(**kwargs):
            value = signals.get((kwargs["symbol"], common.utc(kwargs["now"]).value))
            return TradeSetup.model_validate(value) if value is not None else None
        schedule = {symbol: [pd.Timestamp(nanos, tz="UTC") for item_symbol, nanos in signals if item_symbol == symbol] for symbol in symbols}
        for stress in ((False, True) if phase == "final" else (False,)):
            costs = market["stress_costs_bps_per_side" if stress else "baseline_costs_bps_per_side"]
            cost_settings = settings.model_copy(update={"paper_fee_bps": costs[0], "paper_slippage_bps": costs[1]})
            started = time.monotonic()
            result = portfolio_replay({symbol: frames[symbol] for symbol in symbols}, cost_settings, symbols=symbols,
                start_at=window["start_utc"], end_at=window["end_utc"],
                execution_frames={symbol: frames[symbol][plan["execution_resolution"]] for symbol in symbols},
                funding_rates={symbol: funding[symbol] for symbol in symbols}, signal_provider=provider,
                execution_timeframe=plan["execution_resolution"], signal_schedule=schedule,
                position_limits={symbol: plan["position_limits"][symbol] for symbol in symbols if symbol in plan["position_limits"]})
            result.update({"candidate": candidate, "phase": phase, "market": market["name"],
                           "source_venue": plan["source_venue"], "cost_variant": "stress" if stress else "baseline",
                           "requested_window": window, "elapsed_seconds": round(time.monotonic()-started, 3)})
            payload["results"].append(common.annotate(result))
            common.atomic_json(checkpoint, payload)
            print(f"post-sweep {phase} {market['name']}/{mode} {candidate['id']} {result['cost_variant']}: "
                  f"n={result['trades_count']} TP={result['tp_hits']} net={result['net_pnl_usdt']:.2f}", flush=True)
    return payload




def write_post_sweep_report(out_dir, plan):
    common.write_report(out_dir, plan)
    path = Path(out_dir)/"research_report.md"
    report = path.read_text(encoding="utf-8")
    report = report.replace("24 основных варианта, 6 семантических ablation и v1 зарегистрированы до поиска.",
                            "24 post_sweep_break варианта и три фиксированных контроля зарегистрированы до поиска.")
    prior = plan["prior_recorded_research_trials"] + plan.get("prior_v1_suite_trials", 255)
    prefix = ("**Последняя объявленная контекстная попытка v4:** исходы train/validation уже изучались v2/v3. "
              f"Ранее записано {prior} stage trials; два старых shared-wallet портфеля раскрыты отдельно. "
              "Повторное использование данных не является независимым подтверждением.\n\n")
    prefix += "До новых исходов объявлены оба плана MEXC M5 и Binance M1: общий бюджет81train/18validation/24final. "
    prefix += "Выбор между площадками не переносится; Binance прибыль не является MEXC результатом.\n\n"
    prefix += "C — только закрытый D1 непосредственно после A с HH/HL и close>A.high (SHORT зеркально). "
    prefix += "Этот post-A break может быть против глобального тренда автора; v4 не является полной репликой стратегии.\n\n"
    prefix += "Публичный BTC пример24июня2025 внутри Binance validation был причинно просмотрен до v4; "
    prefix += "его опубликованный исход известен. Validation не объявляется нетронутым.\n\n"
    results = []
    for phase in ("train","validation","final"):
        stage = Path(out_dir)/f"{phase}_results.json"
        if stage.exists():
            results.extend(json.loads(stage.read_text(encoding="utf-8"))["results"])
    report = report.replace("| Кандидат | Издержки |", "| Кандидат | Provider/context | Издержки |")
    report = report.replace("|---|---|---|---|---:|---:|---:|---:|", "|---|---|---|---|---|---:|---:|---:|---:|")
    for row in results:
        provider = row["candidate"]["provider"]
        context = (row["candidate"]["parameters"] or {}).get("context_mode", "v1")
        row_prefix = f"| {row['phase']} | {row['market']}/{row['mode']} | {row['candidate']['id']} | "
        old = row_prefix+row["cost_variant"]+" | "
        report = report.replace(old,row_prefix+provider+"/"+context+" | "+row["cost_variant"]+" | ",1)
    csv_path = Path(out_dir)/"research_summary.csv"
    if results:
        table = pd.read_csv(csv_path,encoding="utf-8-sig")
        table["provider_version"] = [plan["provider_specs"][row["candidate"]["provider"]]["version"] for row in results]
        table["context_mode"] = [(row["candidate"]["parameters"] or {}).get("context_mode","v1") for row in results]
        table["parameters_json"] = [common.canonical(row["candidate"]["parameters"]) for row in results]
        table.to_csv(csv_path,index=False,encoding="utf-8-sig")
    prefix += "Контроли: замороженная v1 с известным ограничением повторных raid; V3 daily/local default с прежними параметрами. "
    prefix += "Контроли не участвуют в выборе кандидата. Все нулевые и отрицательные результаты сохранены.\n\n"
    path.write_text(prefix+report, encoding="utf-8")


def run_post_sweep_phase(out_dir, phase, workers=2):
    out_dir = Path(out_dir)
    plan = json.loads((out_dir/"research_plan.json").read_text(encoding="utf-8"))
    if plan.get("runner") != "app.research_post_sweep":
        raise ValueError("Use the original runner for an original plan")
    verify_post_sweep_plan(plan)
    verify_registered_pair(plan)
    if "registered_protocol_sha256" in plan and common.file_hash(out_dir/"registered_protocol.md") != plan["registered_protocol_sha256"]:
        raise ValueError("Registered source protocol changed")
    output = out_dir/f"{phase}_results.json"
    training = common.load_stage(out_dir, "train", plan)["results"] if phase == "validation" else None
    selection = None
    if phase == "final":
        selection = json.loads((out_dir/"frozen_selection.json").read_text(encoding="utf-8"))
        common.validate_selection(plan, out_dir, selection)
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved.get("completed_at_utc"):
            if saved["plan_sha256"] != plan["plan_sha256"]:
                raise ValueError("Stage belongs to another registered plan")
            if phase == "validation":
                common.freeze_selection(plan, out_dir, training, saved["results"])
            write_post_sweep_report(out_dir, plan)
            return saved
    if phase == "train" and (out_dir/"frozen_selection.json").exists():
        raise ValueError("Cannot reopen train after selection freeze")
    jobs = stage_jobs(plan, phase, training, selection)
    if phase == "final" and jobs:
        common.atomic_json(out_dir/"final_opened.json", {"plan_sha256":plan["plan_sha256"],"selection_sha256":selection["selection_sha256"]})
    expected = {(market["name"], mode, candidate["id"], cost) for market, mode, candidates in jobs for candidate in candidates
                for cost in (("baseline", "stress") if phase == "final" else ("baseline",))}
    checkpoints = out_dir/"work"/phase
    checkpoints.mkdir(parents=True, exist_ok=True)
    payload = {"phase":phase,"plan_sha256":plan["plan_sha256"],"started_at_utc":datetime.now(timezone.utc).isoformat(),
               "expected_trials":len(expected),"results":[],"signal_cache":{}}
    authoritative = []
    with ProcessPoolExecutor(max_workers=max(1,min(workers,len(jobs) or 1))) as pool:
        pending = {pool.submit(run_post_sweep_group, plan, phase, market, mode, candidates,
                               str(checkpoints/f"{market['name']}_{mode}.json")) for market, mode, candidates in jobs}
        previous_count = -1
        while pending:
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            authoritative.extend(future.result() for future in done)
            progress = []
            for market, mode, _ in jobs:
                path = checkpoints/f"{market['name']}_{mode}.json"
                if path.exists():
                    checkpoint = json.loads(path.read_text(encoding="utf-8"))
                    if checkpoint["plan_sha256"] != plan["plan_sha256"]:
                        raise ValueError("Checkpoint belongs to another plan")
                    progress.extend(checkpoint["results"])
            if len(progress) != previous_count:
                payload["results"] = progress
                common.atomic_json(output, payload)
                write_post_sweep_report(out_dir, plan)
                print(f"post-sweep {phase} {len(progress)}/{len(expected)}", flush=True)
                previous_count = len(progress)
    cases = {}
    for value in authoritative:
        payload["signal_cache"][value["market"]+"/"+value["mode"]] = value["signal_cache"]
        for result in value["results"]:
            identity = common.result_identity(result)
            if identity in cases:
                raise ValueError("Duplicate completed post-sweep trial")
            cases[identity] = result
    if set(cases) != expected:
        raise ValueError("Completed trials do not match the preregistered finite stage")
    payload["results"] = list(cases.values())
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    common.atomic_json(output, payload)
    if phase == "validation":
        common.freeze_selection(plan, out_dir, training, payload["results"])
    write_post_sweep_report(out_dir, plan)
    print(f"Completed post-sweep {phase}: {len(expected)} trials", flush=True)
    return payload


def verify_registered_pair(plan):
    """Read peer registration only, never select using its trial outcomes."""
    path = Path(plan["paired_plan_directory"])/"research_plan.json"
    if not path.is_file():
        raise ValueError("Register both predeclared venue plans before any V4 outcomes")
    peer = json.loads(path.read_text(encoding="utf-8"))
    common.verify_plan(peer)
    if (peer.get("runner") != "app.research_post_sweep"
            or {plan["source_venue"], peer["source_venue"]} != {"mexc", "binance_usdm"}
            or peer["candidates"] != plan["candidates"] or peer["code_sha256"] != plan["code_sha256"]
            or peer["joint_post_sweep_budget"] != plan["joint_post_sweep_budget"]
            or peer["parent_trials"] != plan["parent_trials"]
            or peer.get("prior_source_archive_sha256") != plan.get("prior_source_archive_sha256")
            or peer.get("source_review_sha256") != plan.get("source_review_sha256")
            or peer.get("registered_protocol_sha256") != plan.get("registered_protocol_sha256")):
        raise ValueError("Paired registration differs from the declared joint V4 protocol")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("draft", "register", "train", "validation", "final"))
    parser.add_argument("--out-dir", default="outputs/research_post_sweep")
    parser.add_argument("--cache")
    parser.add_argument("--name", default="volium-v4-post-sweep-break")
    parser.add_argument("--market", action="append", help="name=SYMBOL1,SYMBOL2")
    parser.add_argument("--market-costs", action="append", help="name=baselineFee,baselineSlip,stressFee,stressSlip")
    parser.add_argument("--mode", action="append", choices=("intraday",))
    parser.add_argument("--execution-resolution", choices=("1m", "5m"), default="5m")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--fold", action="append", help="train|validation|test=START/END")
    parser.add_argument("--parent-dir", action="append")
    parser.add_argument("--source-review")
    parser.add_argument("--source-archive")
    parser.add_argument("--peer-out-dir", help="Other declared venue plan; both registrations are required before train")
    parser.add_argument("--contract-snapshot")
    parser.add_argument("--workers", type=int, choices=(1,2,3), default=2)
    args = parser.parse_args(argv)
    if args.phase == "draft":
        print(json.dumps({"providers":{key:asdict(spec) for key,spec in PROVIDERS.items()},"candidates":post_sweep_grid(),
                          "policy":common.POLICY,"joint_budget":JOINT_POST_SWEEP_BUDGET,"no_search_executed":True},indent=2))
        return
    if args.phase == "register":
        if not all((args.cache,args.market,args.parent_dir,args.source_review,args.source_archive,args.peer_out_dir)):
            parser.error("Registration requires cache, finite markets, all parent history, source review, archive and peer directory")
        markets = [{"name":item.split("=",1)[0],"symbols":item.split("=",1)[1].split(",")} for item in args.market]
        for item in args.market_costs or []:
            name, numbers = item.split("=",1)
            values = [float(value) for value in numbers.split(",")]
            if len(values) != 4 or name not in {market["name"] for market in markets}:
                parser.error("Market costs require a declared market and four bps values")
            for market in markets:
                if market["name"] == name:
                    market.update(baseline_costs_bps_per_side=values[:2],stress_costs_bps_per_side=values[2:])
        if args.fold:
            folds = {}
            for item in args.fold:
                phase, dates = item.split("=",1)
                start, end = dates.split("/",1)
                folds[phase] = {"start_utc":start,"end_utc":end}
        elif args.start and args.end:
            folds = common.chronological_folds(args.start,args.end,args.execution_resolution)
        else:
            parser.error("Explicit chronological fold bounds are required")
        plan = register_post_sweep(args.cache,args.out_dir,name=args.name,markets=markets,modes=args.mode or ["intraday"],
            resolution=args.execution_resolution,folds=folds,parent_directories=args.parent_dir,
            source_review=args.source_review,source_archive=args.source_archive,peer_out_dir=args.peer_out_dir,
            contract_snapshot=args.contract_snapshot)
        print("Registered post-sweep without candidate outcomes: "+plan["plan_sha256"])
    else:
        run_post_sweep_phase(args.out_dir,args.phase,args.workers)


if __name__ == "__main__":
    main()
