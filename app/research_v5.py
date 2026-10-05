"""Fixed V5 source/economics diagnostics. Public frozen CSVs; no API clients.

Register BOTH venues before outcomes. Three fixed models are reported at every
stage, including final; there is no search, ranking, shortlist or promotion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import time
import zipfile

import pandas as pd

from app import research_suite as common
from app.research_followup import parent_trial_ledger
from app.portfolio_replay import portfolio_replay
from app.portfolio_replay_v5 import portfolio_replay_v5, mexc_admission, binance_proxy_admission
from app.replay import required_timeframes, SECONDS
from app.config import Settings
from app.runtime_settings import RuntimeSettings
from app.schemas.setup import TradeSetup
from app.strategy.volium import analyze_volium_from_df, in_volium_session


CORE = ["BTC_USDT", "ETH_USDT"]
ROBUST_FIVE = ["BTC_USDT", "ETH_USDT", "ZEC_USDT", "SOL_USDT", "DOGE_USDT"]
CANDIDATES = [
    {"id":"v5-strict-cost", "category":"fixed_source_model", "provider":"v5", "engine":"v5",
     "parameters":{"liquidity_mode":"strict", "recovery_fraction":1/3, "max_wait_bars":12}},
    {"id":"v5-equal-cost", "category":"fixed_source_model", "provider":"v5", "engine":"v5",
     "parameters":{"liquidity_mode":"equal_clusters", "recovery_fraction":1/3, "max_wait_bars":12}},
    {"id":"v1-legacy3x", "category":"legacy_diagnostic", "provider":"v1", "engine":"frozen_v1",
     "parameters":None},
]
POLICY = {"fixed_models":True, "ranking":False, "selection_or_promotion_on_final":False,
    "baseline_costs_bps_per_side":[5,2], "stress_costs_bps_per_side":[10,5],
    "initial_shared_equity_usdt":1000, "risk_percent":.5, "max_slots":3, "daily_loss_percent":2,
    "pending_ttl_minutes":30, "eligible_baseline_total_closed_min":30,
    "eligible_validation_closed_min":15, "eligible_final_closed_min":15,
    "requires_positive_baseline_folds":["train","validation","final"],
    "requires_positive_stress_folds":["validation","final"],
    "requires_flat_inventory_all_folds":True, "small_sample_per_fold_below":30,
    "historical_eligibility_is_not_production_proof":True,
    "binance_maintenance_margin_rate_proxy":.01, "binance_no_mexc_tick_lot_tiers":True,
    "prior_stage_trials":635, "old_shared_wallets_separate":2,
    "finite_joint_budget":{"train":9,"validation":18,"final":18,"robustness":12,"total":57}}
CODE_FILES = ("research_v5.py", "portfolio_replay_v5.py", "execution/economics.py", "runtime_settings.py",
    "research_suite.py", "research_followup.py", "portfolio_replay.py", "replay.py", "config.py",
    "strategy/volium.py", "strategy/volium_v2.py", "strategy/volium_v2_corrected.py",
    "strategy/volium_v3.py", "strategy/volium_v4.py", "strategy/volium_v5.py",
    "schemas/setup.py", "exchange/client.py")
FROZEN_OLD_FILES = ("portfolio_replay.py", "replay.py", "config.py", "strategy/volium.py",
    "strategy/volium_v2.py", "strategy/volium_v2_corrected.py", "strategy/volium_v3.py", "strategy/volium_v4.py",
    "research_suite.py", "research_followup.py", "research_post_sweep.py", "schemas/setup.py", "exchange/client.py")


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def approved_folds(venue,parents):
    if venue=="mexc":
        boundaries=("2025-11-08T15:25:00Z","2026-04-07T15:20:00Z",
                    "2026-07-06T15:20:00Z","2026-10-04T15:20:00Z")
        return common.validate_folds({phase:{"start_utc":begin,"end_utc":end}
            for phase,begin,end in zip(("train","validation","test"),boundaries,boundaries[1:])},"5m")
    return common.validate_folds(parents[-1]["folds"],"1m")


def runtime_settings(mode, *, stress=False, profile="v5_strict"):
    values = common.settings_for(mode,stress=stress).model_dump()
    values.update(default_leverage=50,volium_strategy_profile=profile)
    return RuntimeSettings(_env_file=None,**values)


def history_requirements(mode, lookback=80):
    # Full entry-TF prefix supports the actual first raid, not a truncated wick.
    return {"1d":lookback+30,"1h":lookback+30,"5m":lookback*12+30} if mode=="intraday" else {
        "1h":lookback+30,"5m":lookback+30,"1m":lookback*5+30}


def archive_provenance(archive, manifest_path):
    archive,manifest_path=Path(archive).resolve(),Path(manifest_path).resolve()
    metadata=json.loads(manifest_path.read_text(encoding="utf-8"))
    if common.file_hash(archive)!=metadata["sha256"]:
        raise ValueError("Preserved old source archive hash mismatch")
    indexed={item["path"]:item for item in metadata["files"]}
    with zipfile.ZipFile(archive) as capsule:
        names=[item.filename for item in capsule.infolist() if not item.is_dir()]
        if len(names)!=len(set(names)) or set(names)!=set(indexed):
            raise ValueError("Source archive has missing or duplicate files")
        for name in names:
            if hashlib.sha256(capsule.read(name)).hexdigest()!=indexed[name]["sha256"]:
                raise ValueError("Source archive member hash mismatch")
        for relative in FROZEN_OLD_FILES:
            name=next((name for name in names if name.endswith("/app/"+relative)),None)
            if name is None or indexed[name]["sha256"]!=common.file_hash(Path(__file__).parent/relative):
                raise ValueError("Frozen prior source changed: "+relative)
    return {"archive":str(archive),"sha256":metadata["sha256"],"manifest":str(manifest_path),
            "manifest_sha256":common.file_hash(manifest_path),"members":len(indexed)}


def prior_ledger(directories):
    ledger=parent_trial_ledger(directories)
    if len(ledger)!=6 or len({row["plan_sha256"] for row in ledger})!=6:
        raise ValueError("V5 requires all six distinct completed V2/V3/V4 venue plans")
    if any(set(row["phases"])!={"train","validation","final"} for row in ledger):
        raise ValueError("All recorded prior phases must be complete")
    if any(row["final_test_opened"] for row in ledger):
        raise ValueError("Declared old research final periods were not previously opened")
    trials=sum(stage["trials"] for row in ledger for stage in row["phases"].values())
    if trials+255!=POLICY["prior_stage_trials"]:
        raise ValueError("Prior trial ledger differs from the declared 635-stage-trial history")
    return ledger


def data_spec(cache, symbols, modes, resolution, folds):
    cache=Path(cache).resolve()
    manifest=json.loads((cache/"snapshot.json").read_text(encoding="utf-8"))
    if manifest.get("status","complete")!="complete":raise ValueError("Data cache is incomplete")
    folds=common.validate_folds(folds,resolution)
    if common.utc(folds["test"]["end_utc"])>common.utc(manifest["server_snapshot_utc"]):
        raise ValueError("Research ends after the source cutoff")
    paths=common.source_paths(cache,manifest,symbols,modes,resolution)
    hashes={}
    for name,(path,expected) in paths.items():
        if common.file_hash(path)!=expected:raise ValueError("Public data hash mismatch: "+name)
        hashes[name]=expected
    spec={"cache_path":str(cache),"data_manifest_sha256":common.file_hash(cache/"snapshot.json"),
        "data_sha256":hashes,"markets":[{"name":"crypto_core","symbols":symbols}],"modes":modes,
        "source_venue":manifest.get("venue","mexc"),"execution_resolution":resolution,"folds":folds,
        "source_snapshot_utc":manifest["server_snapshot_utc"]}
    # Only timestamp/coverage validation; no signals, price returns or PnL here.
    frames,_=common.read_dataset(spec)
    begin,end=common.utc(folds["train"]["start_utc"]),common.utc(folds["test"]["end_utc"])
    expected=pd.date_range(begin,end-pd.Timedelta(seconds=SECONDS[resolution]),freq=f"{SECONDS[resolution]}s")
    spec["coverage"]={}
    for symbol in symbols:
        frame=frames[symbol][resolution]
        if not frame.index.equals(frame.index.floor(f"{SECONDS[resolution]}s")):
            raise ValueError("Off-grid public execution frame")
        missing=int((frame.index.get_indexer(expected)<0).sum())
        if missing:raise ValueError(f"Missing {missing} execution bars: {symbol}")
        spec["coverage"][symbol]={"execution_bars":len(expected),"missing_bars":missing}
    return spec


def register(cache,out_dir,*,venue,parent_directories,peer_dir,source_review,archive,archive_manifest,
             contract_snapshot,robust_cache=None):
    out_dir=Path(out_dir).resolve()
    if (out_dir/"research_plan.json").exists():raise ValueError("Never overwrite an existing registered plan")
    if venue not in {"mexc","binance_usdm"}:raise ValueError("Unapproved research venue")
    parents=prior_ledger(parent_directories)
    old=[json.loads((Path(row["directory"])/"research_plan.json").read_text(encoding="utf-8"))
         for row in parents if row["source_venue"]==venue]
    folds=approved_folds(venue,old)
    modes=["intraday"] if venue=="mexc" else ["intraday","scalp"]
    resolution="5m" if venue=="mexc" else "1m"
    dataset=data_spec(cache,CORE,modes,resolution,folds)
    if dataset["source_venue"]!=venue:raise ValueError("Data venue differs from the explicit research venue")
    provider=importlib.import_module("app.strategy.volium_v5")
    for name in ("V5Parameters","analyze_volium_v5_batch_from_df"):
        if not callable(getattr(provider,name,None)):raise ValueError("V5 provider API missing")
    for candidate in CANDIDATES[:2]:provider.V5Parameters(**candidate["parameters"])
    contract_path=Path(contract_snapshot).resolve()
    artifact=json.loads(contract_path.read_text(encoding="utf-8"))
    contracts={((record["symbol"][:-4]+"_USDT") if venue=="binance_usdm" else record["symbol"]):record
               for record in artifact["records"]}
    if not set(CORE).issubset(contracts):raise ValueError("Contract proxy snapshot lacks BTC/ETH")
    archive_record=archive_provenance(archive,archive_manifest)
    root=Path(__file__).parent
    protocol=root.parent/"docs/research_v5_protocol.md"
    source_review=Path(source_review).resolve()
    plan={**dataset,"version":5,"runner":"app.research_v5","experiment":"fixed-v5-source-and-cost-"+venue,
        "registered_at_utc":now_utc(),"peer_directory":str(Path(peer_dir).resolve()),"candidates":CANDIDATES,
        "policy":POLICY,"parent_trials":parents,"preserved_prior_source":archive_record,
        "source_review":{"path":str(source_review),"sha256":common.file_hash(source_review)},
        "protocol_path":str(protocol.resolve()),"protocol_sha256":common.file_hash(protocol),
        "code_sha256":{name:common.file_hash(root/name) for name in CODE_FILES},
        "code_hash_scope":"executed strategy, explicit V5/legacy cash models, policy and imported pure helpers",
        "contracts":contracts,"contract_snapshot":{"path":str(contract_path),"sha256":common.file_hash(contract_path)},
        "runtime_settings_by_mode":{mode:runtime_settings(mode).model_dump(mode="json") for mode in modes},
        "legacy_settings_by_mode":{mode:common.settings_for(mode).model_dump(mode="json") for mode in modes},
        "history_bars_by_mode":{mode:history_requirements(mode) for mode in modes},
        "cross_venue_experiment":venue!="mexc","never_claimed_as_mexc_pnl":venue!="mexc",
        "holdout_is_not_guaranteed_pristine":True,
        "previously_inspected":["635 previous stage trials and two shared-wallet runs",
            "current core2 and current originalfive retrospective market hypotheses",
            "MEXC full330d folds extended before V5 registration to150/90/90days",
            "old MEXC train/validation reused; old unopened Mar-Apr final now in new train",
            "later MEXC180d context/individual experiments and last30d portfolios already inspected",
            "BTC June24 2025 validation causal features and public author outcome",
            "previous source hypotheses and source-only train cases",
            "new video/source transfer; exact author chart classification unavailable"],
        "robustness_data":None}
    if robust_cache is not None:
        if venue!="mexc":raise ValueError("MEXC originalfive robustness cannot be mixed with Binance")
        manifest=json.loads((Path(robust_cache)/"snapshot.json").read_text(encoding="utf-8"))
        end=common.utc(manifest["server_snapshot_utc"]).floor("min")
        begin=end-pd.Timedelta(days=30)+pd.Timedelta(minutes=1)
        # This known window is one descriptive run, not split into pretend holdouts.
        artificial=common.chronological_folds(begin,end,"1m")
        robust=data_spec(robust_cache,ROBUST_FIVE,["intraday","scalp"],"1m",artificial)
        robust["window"]={"start_utc":begin.isoformat(),"end_utc":end.isoformat()}
        robust["not_independent_holdout"]=True
        plan["robustness_data"]=robust
        if not set(ROBUST_FIVE).issubset(contracts):raise ValueError("Contract proxy snapshot lacks predeclared core+top3 five")
    elif venue=="mexc":
        raise ValueError("The approved57-trial protocol includes originalfive robustness")
    peer_fingerprint={"code_sha256":plan["code_sha256"],"candidates":CANDIDATES,"policy":POLICY,
        "parent_trials":parents,"source_review":plan["source_review"],"preserved_prior_source":archive_record,
        "protocol_sha256":plan["protocol_sha256"]}
    plan["joint_policy_sha256"]=common.digest(peer_fingerprint)
    plan["plan_sha256"]=common.digest(plan)
    common.atomic_json(out_dir/"research_plan.json",plan)
    (out_dir/"registered_protocol.md").write_bytes(protocol.read_bytes())
    return plan


def verify(plan):
    common.verify_plan(plan)
    source=plan["source_review"]
    if common.file_hash(source["path"])!=source["sha256"]:raise ValueError("Source review changed after registration")
    if common.file_hash(plan["protocol_path"])!=plan["protocol_sha256"]:
        raise ValueError("Protocol changed after registration")
    capsule=plan["preserved_prior_source"]
    if common.file_hash(capsule["archive"])!=capsule["sha256"] or common.file_hash(capsule["manifest"])!=capsule["manifest_sha256"]:
        raise ValueError("Preserved prior source capsule changed")
    if common.file_hash(plan["contract_snapshot"]["path"])!=plan["contract_snapshot"]["sha256"]:
        raise ValueError("Financial proxy metadata changed")
    for parent in plan["parent_trials"]:
        directory=Path(parent["directory"])
        if common.file_hash(directory/"research_plan.json")!=parent["plan_file_sha256"]:raise ValueError("Parent plan changed")
        for phase,metadata in parent["phases"].items():
            if common.file_hash(directory/f"{phase}_results.json")!=metadata["sha256"]:raise ValueError("Parent results changed")
    peer=json.loads((Path(plan["peer_directory"])/"research_plan.json").read_text(encoding="utf-8"))
    common.verify_plan(peer)
    if peer["runner"]!=plan["runner"] or peer["source_venue"]==plan["source_venue"] or peer["joint_policy_sha256"]!=plan["joint_policy_sha256"]:
        raise ValueError("Register both approved venues with the identical fixed policy before outcomes")


def stage_jobs(plan,phase):
    if phase=="robustness":
        return [("core_plus_top3",mode,True) for mode in ("intraday","scalp")] if plan["robustness_data"] else []
    if phase not in {"train","validation","final"}:raise ValueError("Unknown research stage")
    return [("crypto_core",mode,phase!="train") for mode in plan["modes"]]


def expected_identities(plan,phase):
    return {(market,mode,candidate["id"],cost) for market,mode,stress in stage_jobs(plan,phase)
            for candidate in CANDIDATES for cost in (("baseline","stress") if stress else ("baseline",))}


def collect_complete(payloads,expected,phase=None):
    cases={}
    for payload in payloads:
        for row in payload["results"]:
            if phase is not None and row["phase"]!=phase:
                raise ValueError("Completed result belongs to another phase")
            identity=(row["market"],row["mode"],row["candidate"]["id"],row["cost_variant"])
            if identity in cases:raise ValueError("Duplicate completed fixed-model trial")
            cases[identity]=row
    if set(cases)!=expected:raise ValueError("Completed trials differ from the registered fixed budget")
    return list(cases.values())


def precompute(frames,symbols,settings,window,history_bars):
    module=importlib.import_module("app.strategy.volium_v5")
    parameters=[module.V5Parameters(**candidate["parameters"]) for candidate in CANDIDATES[:2]]
    tables={candidate["id"]:{} for candidate in CANDIDATES}
    tfs=required_timeframes(settings)
    begin,end=common.utc(window["start_utc"]),common.utc(window["end_utc"])
    checks=0
    for symbol in symbols:
        close_indices={tf:frames[symbol][tf].index+pd.Timedelta(seconds=SECONDS[tf]) for tf in tfs}
        times=close_indices[tfs[-1]]
        times=times[(times>begin)&(times<=end)]
        for index,now in enumerate(times,1):
            if index%10000==0:print(f"V5 features {settings.volium_mode} {symbol} {index}/{len(times)}",flush=True)
            if settings.volium_session_enabled and not in_volium_session(now,settings):continue
            known={tf:frames[symbol][tf].iloc[:close_indices[tf].searchsorted(now,side="right")].tail(history_bars[tf]) for tf in tfs}
            kwargs={"symbol":symbol,"frames":known,"settings":settings,"mode":settings.volium_mode,"now":now.to_pydatetime()}
            signals=module.analyze_volium_v5_batch_from_df(parameter_sets=parameters,**kwargs)
            if len(signals)!=2:raise ValueError("V5 fixed batch length mismatch")
            legacy_known={tf:frame.tail(settings.volium_context_lookback+30) for tf,frame in known.items()}
            legacy=analyze_volium_from_df(**{**kwargs,"frames":legacy_known})
            for candidate,signal in zip(CANDIDATES,[*signals,legacy]):
                if signal is not None:
                    if signal.symbol!=symbol or common.utc(signal.timestamp)!=now:raise ValueError("Provider output lacks causal signal identity")
                    tables[candidate["id"]][(symbol,now.value)]=signal.model_dump(mode="json")
            checks+=1
    return tables,{"session_confirmation_checks":checks,
                   "positive_outputs_by_fixed_model":{key:len(table) for key,table in tables.items()}}


def run_group(plan,phase,market,mode,with_stress,checkpoint):
    verify(plan)
    spec=plan["robustness_data"] if phase=="robustness" else plan
    window=spec["window"] if phase=="robustness" else plan["folds"]["test" if phase=="final" else phase]
    symbols=ROBUST_FIVE if phase=="robustness" else CORE
    frames,funding=common.read_dataset(spec,end_at=window["end_utc"])
    settings=RuntimeSettings(_env_file=None,**plan["runtime_settings_by_mode"].get(mode,runtime_settings(mode).model_dump()))
    history=plan["history_bars_by_mode"].get(mode,history_requirements(mode))
    tables,metrics=precompute(frames,symbols,settings,window,history)
    payload={"phase":phase,"market":market,"mode":mode,"plan_sha256":plan["plan_sha256"],"results":[],"signal_cache":metrics}
    for candidate in CANDIDATES:
        signals=tables[candidate["id"]]
        def provider(**kwargs):
            value=signals.get((kwargs["symbol"],common.utc(kwargs["now"]).value))
            return TradeSetup.model_validate(value) if value is not None else None
        schedule={symbol:[pd.Timestamp(nanos,tz="UTC") for key,nanos in signals if key==symbol] for symbol in symbols}
        for stress in ((False,True) if with_stress else (False,)):
            started=time.monotonic()
            if candidate["engine"]=="frozen_v1":
                policy=common.settings_for(mode,stress=stress)
                result=portfolio_replay({symbol:frames[symbol] for symbol in symbols},policy,symbols=symbols,
                    start_at=window["start_utc"],end_at=window["end_utc"],funding_rates=funding,
                    signal_provider=provider,execution_timeframe=spec["execution_resolution"],signal_schedule=schedule)
                result["engine_version"]="frozen_v1_legacy3x"
            else:
                policy=runtime_settings(mode,stress=stress,profile="v5_equal" if candidate["parameters"]["liquidity_mode"]=="equal_clusters" else "v5_strict")
                admission=mexc_admission(plan["contracts"]) if plan["source_venue"]=="mexc" else binance_proxy_admission(plan["contracts"])
                result=portfolio_replay_v5({symbol:frames[symbol] for symbol in symbols},policy,symbols=symbols,
                    start_at=window["start_utc"],end_at=window["end_utc"],funding_rates=funding,
                    signal_provider=provider,execution_timeframe=spec["execution_resolution"],signal_schedule=schedule,
                    admission=admission,history_bars=history)
            result.update(candidate=candidate,phase=phase,market=market,source_venue=plan["source_venue"],
                cost_variant="stress" if stress else "baseline",requested_window=window,
                elapsed_seconds=round(time.monotonic()-started,3))
            payload["results"].append(common.annotate(result))
            common.atomic_json(checkpoint,payload)
            print(f"V5 {phase} {market}/{mode} {candidate['id']} {result['cost_variant']}: n={result['trades_count']} TP={result['tp_hits']} net={result['net_pnl_usdt']:.3f}",flush=True)
    return payload


def eligibility(rows,mode,candidate_id):
    selected={(row["phase"],row["cost_variant"]):row for row in rows if row["market"]=="crypto_core"
              and row["mode"]==mode and row["candidate"]["id"]==candidate_id}
    required={(phase,"baseline") for phase in ("train","validation","final")}|{(phase,"stress") for phase in ("validation","final")}
    if not required.issubset(selected):return {"eligible":False,"verdict":"incomplete_fixed_diagnostic"}
    baseline=[selected[(phase,"baseline")] for phase in ("train","validation","final")]
    total=sum(row["trades_count"] for row in baseline)
    reasons=[]
    if candidate_id=="v1-legacy3x":reasons.append("legacy_control_not_candidate")
    if total<30:reasons.append("fewer_than_30_total_closed")
    for phase in ("validation","final"):
        if selected[(phase,"baseline")]["trades_count"]<15:reasons.append(phase+"_fewer_than_15_closed")
    for key in sorted(required):
        row=selected[key]
        if row["net_pnl_usdt"]<=0:reasons.append("_".join(key)+"_nonpositive")
        if row["unfinished_positions"] or row["unfinished_limits"]:reasons.append("_".join(key)+"_unfinished_inventory")
    return {"eligible":not reasons,"verdict":"exploratory_positive_historical_candidate" if not reasons else "no_historical_eligibility",
        "reasons":reasons,"baseline_total_closed":total,"small_samples":[row["phase"] for row in baseline if row["trades_count"]<30],
        "production_edge_proven":False}


def write_report(out_dir,plan):
    out_dir=Path(out_dir); rows=[]
    for phase in ("train","validation","final","robustness"):
        path=out_dir/f"{phase}_results.json"
        if path.exists():rows.extend(json.loads(path.read_text(encoding="utf-8"))["results"])
    title="MEXC" if plan["source_venue"]=="mexc" else "Binance USDT perpetual — отдельная площадка"
    lines=[f"# V5: фиксированная проверка — {title}","",
        "Три заранее зафиксированных модели; подбор победителя и перенос выводов между площадками запрещены. ",
        "Исходные контекстные численные прокси v1 сохранены. Две новые модели одинаково исправляют события снятия, H1-согласование и вход через engulfment/инверсию FVG; различается только строгая/равная ликвидность.","",
        "Ранее просмотрены 635 stage trials и отдельно два общих портфеля; прежние train/validation переиспользуются. Пример автора BTC 24 июня 2025 известен внутри Binance validation. MEXC до регистрации расширен на все 330 дней; старый unopened final входит в новый train, поздние окна частично уже изучены. Здесь все фиксированные final модели оцениваются один раз, даже при нуле сигналов. Это не гарантированно нетронутая независимая проверка.","",
        f"План SHA256: {plan['plan_sha256']}. Общий бюджет двух площадок: 57 новых trials; каждый режим отдельно начинает с общего счёта 1000 USDT.","",
        "| Этап | Режим/рынок | Модель | Издержки | Закрытых | TP/нетто побед | PnL | Funding | Издержки cash | Открыто |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        costs=row.get("fees_total_usdt",0)+row.get("slippage_total_usdt",0)
        cost_cell=f"{costs:.4f}" if row["candidate"]["engine"]=="v5" else "old flat model"
        lines.append(f"| {row['phase']} | {row['mode']}/{row['market']} | {row['candidate']['id']} | {row['cost_variant']} | {row['trades_count']} | {row['tp_hits']}/{row['wins']} | {row['net_pnl_usdt']:.3f} | {row['funding_total_usdt']:.4f} | {cost_cell} | {row['unfinished_positions']}/{row['unfinished_limits']} |")
    verdicts={mode:{candidate["id"]:eligibility(rows,mode,candidate["id"]) for candidate in CANDIDATES} for mode in plan["modes"]}
    lines.extend(["","Историческая eligibility требует положительногоbaseline train/validation/final, положительногоstress validation/final, минимум30закрытых всего и15отдельно вvalidation иfinal, без незавершённогоинвентаря. Ни один результат не является автоматическим разрешением live.","",
        "Комиссия и проскальзывание новой модели списываются при заполнении входа и при выходе по соответствующему номиналу; реальный funding своей площадки учтён один раз. Плечо 10–50 меняет маржу, а плановый риск остаётся до 0,5% с учётом издержек. Gap и funding могут превысить плановый риск. Маржа новых заявок ограничена 90% минимума cash/marked equity с учётом других резервов.","",
        "Текущие контрактные ограничения MEXC — исторический прокси. Для Binance MM=1% — отдельная инженерная гипотеза, публичная liquidation fee учтена; MEXC tick/lot/tiers не перенесены. Distance guard не реконструирует исторические ликвидации или mark price. OHLC не описывает очередь лимитов, частичные fills и spread; SL priority и отказ от TP на неопределённой fill-свече сохраняются. M5 явно обозначен как coarse proxy.","",
        "core_plus_top3 robustness (BTC, ETH, ZEC, SOL, DOGE) использует уже известные 30 дней и текущий ретроспективный отбор трёх volatile пар с обязательным core; он исключён из eligibility. BTC/ETH рынок также предложен после прошлых результатов: selection/survivorship bias остаётся.","","Фиксированные вердикты:",""])
    for mode,items in verdicts.items():
        for identifier,value in items.items():lines.append(f"- {mode}/{identifier}: {value['verdict']}; "+", ".join(value.get("reasons",[])))
    (out_dir/"research_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    common.atomic_json(out_dir/"fixed_model_verdicts.json",{"plan_sha256":plan["plan_sha256"],"verdicts":verdicts})
    if rows:
        fields=("phase","market","mode","cost_variant","trades_count","tp_hits","wins","net_pnl_usdt",
                "funding_total_usdt","marked_equity","max_marked_drawdown_percent","unfinished_positions","unfinished_limits",
                "unfilled_signals","long_trades","short_trades","signals","accepted_signals")
        table=[{**{field:row[field] for field in fields},"model":row["candidate"]["id"],
                "fees_usdt":row.get("fees_total_usdt"),"slippage_usdt":row.get("slippage_total_usdt"),
                "rejected_admission":row.get("rejected_admission"),"net_win_wilson95":common.canonical(row["net_win_wilson_95_percent"])} for row in rows]
        pd.DataFrame(table).to_csv(out_dir/"research_summary.csv",index=False,encoding="utf-8-sig")


def run_phase(out_dir,phase,workers=2):
    out_dir=Path(out_dir)
    plan=json.loads((out_dir/"research_plan.json").read_text(encoding="utf-8")); verify(plan)
    if plan["runner"]!="app.research_v5":raise ValueError("Use the matching registered runner")
    for prerequisite in (("train",) if phase=="validation" else (("train","validation") if phase=="final" else ())):
        path=out_dir/f"{prerequisite}_results.json"
        if not path.exists():raise ValueError("Complete chronological earlier stages before final")
        stage=json.loads(path.read_text(encoding="utf-8"))
        if not stage.get("completed_at_utc") or stage["plan_sha256"]!=plan["plan_sha256"]:raise ValueError("Incomplete prerequisite stage")
        collect_complete([stage],expected_identities(plan,prerequisite),prerequisite)
    output=out_dir/f"{phase}_results.json"; expected=expected_identities(plan,phase)
    if output.exists():
        saved=json.loads(output.read_text(encoding="utf-8"))
        if saved.get("completed_at_utc"):
            if saved["plan_sha256"]!=plan["plan_sha256"]:raise ValueError("Stage plan mismatch")
            collect_complete([saved],expected,phase);write_report(out_dir,plan);return saved
    if phase in {"train","validation"} and (out_dir/"final_opened.json").exists():
        raise ValueError("Cannot reopen earlier unfinished stages after final inspection")
    if phase=="final":
        marker=out_dir/"final_opened.json"
        if marker.exists():
            value=json.loads(marker.read_text(encoding="utf-8"))
            if value["plan_sha256"]!=plan["plan_sha256"]:raise ValueError("Final already belongs to another plan")
            if any(value.get(earlier+"_results_sha256")!=common.file_hash(out_dir/f"{earlier}_results.json") for earlier in ("train","validation")):
                raise ValueError("Earlier completed outcomes changed after final opened")
        else:common.atomic_json(marker,{"plan_sha256":plan["plan_sha256"],"opened_at_utc":now_utc(),
            "fixed_models_no_selection":True,**{earlier+"_results_sha256":common.file_hash(out_dir/f"{earlier}_results.json")
                                                for earlier in ("train","validation")}})
    jobs=stage_jobs(plan,phase); checkpoints=out_dir/"work"/phase;checkpoints.mkdir(parents=True,exist_ok=True)
    payload={"phase":phase,"plan_sha256":plan["plan_sha256"],"started_at_utc":now_utc(),"expected_trials":len(expected),"results":[]}
    authoritative=[]
    with ProcessPoolExecutor(max_workers=max(1,min(workers,len(jobs) or 1))) as pool:
        pending={pool.submit(run_group,plan,phase,market,mode,stress,str(checkpoints/f"{market}_{mode}.json")) for market,mode,stress in jobs}
        previous=-1
        while pending:
            done,pending=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            authoritative.extend(future.result() for future in done)
            progress=[]
            for market,mode,_ in jobs:
                path=checkpoints/f"{market}_{mode}.json"
                if path.exists():
                    checkpoint=json.loads(path.read_text(encoding="utf-8"))
                    if checkpoint["plan_sha256"]!=plan["plan_sha256"]:raise ValueError("Checkpoint plan mismatch")
                    progress.extend(checkpoint["results"])
            if len(progress)!=previous:
                payload["results"]=progress;common.atomic_json(output,payload);write_report(out_dir,plan)
                print(f"V5 {phase} {len(progress)}/{len(expected)}",flush=True);previous=len(progress)
    payload["results"]=collect_complete(authoritative,expected,phase)
    payload["signal_cache"]={value["market"]+"/"+value["mode"]:value["signal_cache"] for value in authoritative}
    payload["completed_at_utc"]=now_utc();common.atomic_json(output,payload);write_report(out_dir,plan)
    print(f"Completed fixed V5 {phase}: {len(expected)}",flush=True)
    return payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("draft","register","train","validation","final","robustness"))
    parser.add_argument("--out-dir",default="outputs/research_v5")
    parser.add_argument("--venue",choices=("mexc","binance_usdm"))
    parser.add_argument("--cache");parser.add_argument("--robust-cache");parser.add_argument("--peer-dir")
    parser.add_argument("--parent-dir",action="append");parser.add_argument("--source-review")
    parser.add_argument("--source-archive");parser.add_argument("--archive-manifest");parser.add_argument("--contract-snapshot")
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    args=parser.parse_args(argv)
    if args.phase=="draft":
        print(json.dumps({"candidates":CANDIDATES,"policy":POLICY,"no_outcomes_computed":True},indent=2));return
    if args.phase=="register":
        if not all((args.venue,args.cache,args.peer_dir,args.parent_dir,args.source_review,args.source_archive,args.archive_manifest,args.contract_snapshot)):
            parser.error("Registration requires venue/cache/peer, allsixparents, source/sourcearchive/manifest and financialmetadata")
        plan=register(args.cache,args.out_dir,venue=args.venue,parent_directories=args.parent_dir,peer_dir=args.peer_dir,
            source_review=args.source_review,archive=args.source_archive,archive_manifest=args.archive_manifest,
            contract_snapshot=args.contract_snapshot,robust_cache=args.robust_cache)
        print("Registered fixed models without outcomes: "+plan["plan_sha256"])
    else:run_phase(args.out_dir,args.phase,args.workers)


if __name__=="__main__":main()
