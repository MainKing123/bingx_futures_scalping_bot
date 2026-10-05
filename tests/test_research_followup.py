import json
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict

import pandas as pd
import pytest

from app import research_followup as followup
from app import research_suite as original
from app.schemas.setup import TradeSetup


def _row(candidate, n=10, pnl=10, market="core"):
    return {"market":market,"mode":"intraday","candidate":candidate,"cost_variant":"baseline",
            "trades_count":n,"wins":n//2,"net_pnl_usdt":pnl,"return_percent":pnl/10,
            "profit_factor":2,"unfinished_positions":0}


def _plan(markets=("core",)):
    return {"runner":"app.research_followup","plan_sha256":"registered",
            "markets":[{"name":name,"symbols":["BTC_USDT"]} for name in markets],"modes":["intraday"],
            "candidates":followup.followup_grid(),"policy":deepcopy(original.POLICY),
            "prior_recorded_research_trials":64}


def _stage(tmp_path, plan, phase, rows):
    value = {"plan_sha256":plan["plan_sha256"],"completed_at_utc":"2026-10-04T00:00:00Z","results":rows}
    original.atomic_json(tmp_path/f"{phase}_results.json",value)
    return value


def test_followup_is_finite_versioned_and_old_registry_is_unchanged():
    old_grid, old_worker = original.candidate_grid(), original.run_group
    grid = followup.followup_grid()
    assert len(grid) == len({candidate["id"] for candidate in grid}) == 51
    main = [candidate for candidate in grid if candidate["category"] == "candidate"]
    assert len(main) == 48
    assert {candidate["parameters"]["context_mode"] for candidate in main} == set(followup.CONTEXTS)
    assert len({candidate["id"] for candidate in main if candidate["parameters"]["context_mode"] == "latest_leg_daily"}) == 24
    assert {candidate["provider"] for candidate in grid if candidate["category"] == "control"} == {"v1","v2_original","v2_corrected"}
    assert all(candidate["provider"] == "v3" for candidate in main)
    assert old_grid == original.candidate_grid() and old_worker is original.run_group
    assert not {candidate["id"] for candidate in main} & {candidate["id"] for candidate in old_grid}


def test_draft_requires_no_new_provider_module_or_data(capsys):
    followup.main(["draft"])
    value = json.loads(capsys.readouterr().out)
    assert value["no_search_executed"] is True
    assert len(value["candidates"]) == 51


def test_declared_provider_exports_and_every_parameter_cell_bind_without_outcomes():
    for candidate in followup.followup_grid():
        factory,batch,diagnostic,single = followup.provider_functions(followup.PROVIDERS[candidate["provider"]])
        if factory is None:
            assert callable(single)
        else:
            parameters = factory(**candidate["parameters"])
            assert parameters.reaction_atr_period == 14
            assert callable(batch) and callable(diagnostic)


def test_parent_ledger_records_completed_trials_and_rejects_active_phase(tmp_path):
    original.atomic_json(tmp_path/"research_plan.json",{"plan_sha256":"parent","source_venue":"mexc","experiment":"old-v2"})
    original.atomic_json(tmp_path/"train_results.json",{"plan_sha256":"parent","completed_at_utc":"now","results":[{}]*62})
    original.atomic_json(tmp_path/"validation_results.json",{"plan_sha256":"parent","completed_at_utc":"now","results":[{}]*2})
    ledger = followup.parent_trial_ledger([tmp_path])
    assert sum(phase["trials"] for phase in ledger[0]["phases"].values()) == 64
    assert ledger[0]["final_test_opened"] is False
    original.atomic_json(tmp_path/"final_results.json",{"plan_sha256":"parent","results":[]})
    with pytest.raises(ValueError,match="Finish"):
        followup.parent_trial_ledger([tmp_path])


def test_controls_never_select_and_all_fixed_controls_survive_stage_jobs():
    plan = _plan()
    candidate = plan["candidates"][0]
    train = [_row(candidate)] + [_row(control,pnl=100) for control in plan["candidates"] if control["category"] == "control"]
    jobs = followup.stage_jobs(plan,"validation",training=train)
    assert len(jobs[0][2]) == 4
    assert [trial["id"] for trial in jobs[0][2]][0] == candidate["id"]
    no_winner = {"winners":{"core/intraday":None}}
    assert followup.stage_jobs(plan,"final",selection=no_winner) == []
    chosen = {"winners":{"core/intraday":candidate}}
    assert len(followup.stage_jobs(plan,"final",selection=chosen)[0][2]) == 4


def test_batch_receives_causal_extended_h1_prefix_and_all_variants(monkeypatch):
    settings = original.settings_for("intraday").model_copy(update={"volium_session_enabled":False})
    start = pd.Timestamp("2026-01-01T07:00Z")
    frames = {}
    for tf, periods, end in [("1d",600,start+pd.Timedelta(days=200)),("1h",4000,start+pd.Timedelta(days=10)),
                            ("5m",1000,start+pd.Timedelta(hours=1))]:
        index = pd.date_range(end=end,periods=periods,freq=f"{original.SECONDS[tf]}s")
        frames[tf] = pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.},index=index)
    grid = followup.followup_grid()
    candidates = [grid[0],grid[1],next(candidate for candidate in grid if candidate["provider"] == "v2_corrected")]
    calls = []
    def functions(spec):
        def batch(*,parameter_sets,**kwargs):
            now = original.utc(kwargs["now"])
            assert len(kwargs["frames"]["1h"]) == (2030 if spec.version == "experimental_v3" else 110)
            assert len(kwargs["frames"]["5m"]) == 110
            for tf, frame in kwargs["frames"].items():
                assert (frame.index+pd.Timedelta(seconds=original.SECONDS[tf]) <= now).all()
            calls.append((spec.version,len(parameter_sets)))
            return [TradeSetup(id=f"{spec.version}-{i}-{now.value}",timestamp=now.to_pydatetime(),symbol=kwargs["symbol"],
                               direction="LONG",setup_type="VOLIUM_INTRADAY",htf_bias="BULLISH",entry=100,
                               stop_loss=99,take_profits=[102],risk_reward=2,confluences=[]) for i in range(len(parameter_sets))]
        return (lambda **kwargs:kwargs),batch,None,None
    monkeypatch.setattr(followup,"provider_functions",functions)
    window = {"start_utc":start.isoformat(),"end_utc":(start+pd.Timedelta(minutes=15)).isoformat()}
    specs = {name:asdict(spec) for name,spec in followup.PROVIDERS.items()}
    tables,metrics = followup.precompute_followup({"BTC_USDT":frames},["BTC_USDT"],settings,candidates,window,specs)
    assert len(calls) == 6 and metrics["eligible_time_boundaries"] == 3
    assert all(len(table) == 3 for table in tables.values())
    assert metrics["batch_calls_by_provider"] == {"v3":3,"v2_corrected":3}


def test_followup_phase_uses_its_explicit_worker_and_pending_only_wait(tmp_path,monkeypatch):
    plan = _plan(("first","second"))
    plan["candidates"] = plan["candidates"][:1]
    original.atomic_json(tmp_path/"research_plan.json",plan)
    monkeypatch.setattr(original,"verify_plan",lambda value:None)
    monkeypatch.setattr(followup,"verify_followup_plan",lambda value:None)
    monkeypatch.setattr(followup,"write_followup_report",lambda *args:None)
    values, waits = {}, []
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,plan,phase,market,mode,candidates,checkpoint):
            assert fn is followup.run_followup_group
            assert fn is not original.run_group
            future = Future()
            values[future] = {"market":market["name"],"mode":mode,"signal_cache":{},
                              "results":[_row(candidates[0],market=market["name"])]}
            return future
    def waiter(pending,**kwargs):
        pending = set(pending)
        assert not any(future.done() for future in pending)
        waits.append(len(pending))
        future = next(iter(pending))
        future.set_result(values[future])
        return {future},pending-{future}
    monkeypatch.setattr(followup,"ProcessPoolExecutor",Pool)
    monkeypatch.setattr(followup,"wait",waiter)
    result = followup.run_followup_phase(tmp_path,"train",2)
    assert waits == [2,1] and len(result["results"]) == 2
    assert result["completed_at_utc"]


def test_followup_completed_validation_recovers_freeze(tmp_path,monkeypatch):
    plan = _plan()
    candidate = plan["candidates"][0]
    original.atomic_json(tmp_path/"research_plan.json",plan)
    _stage(tmp_path,plan,"train",[_row(candidate)])
    stage = _stage(tmp_path,plan,"validation",[_row(candidate,n=3)])
    monkeypatch.setattr(original,"verify_plan",lambda value:None)
    monkeypatch.setattr(followup,"verify_followup_plan",lambda value:None)
    monkeypatch.setattr(followup,"write_followup_report",lambda *args:None)
    assert followup.run_followup_phase(tmp_path,"validation") == stage
    frozen = json.loads((tmp_path/"frozen_selection.json").read_text(encoding="utf-8"))
    assert frozen["winners"]["core/intraday"] == candidate


def test_original_plan_cannot_be_run_using_followup_runner(tmp_path):
    original.atomic_json(tmp_path/"research_plan.json",{"runner":"app.research_suite"})
    with pytest.raises(ValueError,match="original runner"):
        followup.run_followup_phase(tmp_path,"train")


def test_parent_design_refuses_new_windows_pairs_costs_or_constraints(tmp_path):
    folds = original.chronological_folds("2026-01-01","2026-01-11","5m")
    markets = [{"name":"core","symbols":["BTC_USDT","ETH_USDT"],"baseline_costs_bps_per_side":[5,2],"stress_costs_bps_per_side":[10,5]}]
    parent = {"plan_sha256":"parent","source_venue":"mexc","data_manifest_sha256":"frozen-data","markets":markets,
              "modes":["intraday"],"execution_resolution":"5m","folds":folds,"position_limits":{},
              "settings_by_mode":{"intraday":original.settings_for("intraday").model_dump(mode="json")}}
    original.atomic_json(tmp_path/"research_plan.json",parent)
    design = dict(venue="mexc",manifest_sha256="frozen-data",markets=markets,modes=["intraday"],resolution="5m",folds=folds,contracts={})
    assert followup.enforce_parent_design([tmp_path],**design) == "parent"
    for key,value in [("markets",[{**markets[0],"symbols":["BTC_USDT"]}]),("contracts",{"BTC_USDT":{}}),
                      ("resolution","1m"),("folds",original.chronological_folds("2026-01-02","2026-01-12","5m"))]:
        changed = {**design,key:value}
        with pytest.raises(ValueError,match="differ"):
            followup.enforce_parent_design([tmp_path],**changed)


def test_new_registration_signs_providers_parent_trials_and_history_without_analysis(tmp_path,monkeypatch):
    cache = tmp_path/"data"
    cache.mkdir()
    start = pd.Timestamp("2026-01-01T00:00Z")
    manifest = {"status":"complete","server_snapshot_utc":"2026-01-02T00:00Z",
                "frames":{"BTC_USDT":{}},"funding":{"BTC_USDT":{}}}
    for tf in ("1d","1h","5m"):
        index = pd.date_range(start=start-pd.Timedelta(days=5),end=start+pd.Timedelta(hours=2),freq=f"{original.SECONDS[tf]}s")
        frame = pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.},index=index)
        path = cache/f"BTC_USDT_{tf}.csv"
        frame.to_csv(path)
        manifest["frames"]["BTC_USDT"][tf] = {"sha256":original.file_hash(path)}
    funding = cache/"BTC_USDT_funding.csv"
    pd.Series([.0001],index=pd.DatetimeIndex([start]),name="funding_rate").to_csv(funding)
    manifest["funding"]["BTC_USDT"] = {"sha256":original.file_hash(funding),"status":"available"}
    original.atomic_json(cache/"snapshot.json",manifest)
    markets = [{"name":"core","symbols":["BTC_USDT"]}]
    folds = original.chronological_folds(start,start+pd.Timedelta(minutes=75),"5m")
    parent_dir = tmp_path/"parent"
    parent = original.register_plan(cache,parent_dir,name="synthetic-parent",markets=markets,modes=["intraday"],resolution="5m",folds=folds)
    _stage(parent_dir,parent,"train",[{}]*62)
    _stage(parent_dir,parent,"validation",[{}]*2)
    _stage(parent_dir,parent,"final",[])
    review,archive = tmp_path/"source.md",tmp_path/"old-source.zip"
    review.write_text("Reviewed source hypothesis",encoding="utf-8")
    archive.write_bytes(b"archived source evidence fixture")
    def no_outcomes(spec):
        def must_not_analyze(**kwargs):
            raise AssertionError("Registration must not calculate new candidate outcomes")
        factory = lambda **kwargs:kwargs
        return (factory,must_not_analyze,must_not_analyze,None) if spec.parameter_type else (None,None,None,must_not_analyze)
    monkeypatch.setattr(followup,"provider_functions",no_outcomes)
    out = tmp_path/"followup"
    plan = followup.register_followup(cache,out,name="synthetic-v3",markets=deepcopy(markets),modes=["intraday"],resolution="5m",folds=folds,
                                      parent_directories=[parent_dir],source_review=review,source_archive=archive)
    assert plan["design_parent_plan_sha256"] == parent["plan_sha256"]
    assert plan["prior_recorded_research_trials"] == 64
    assert plan["prior_v1_suite_trials"] == 255
    assert plan["search_budget"] == {"train_main":48,"train_controls":3,"validation_max":6,"final_baseline_and_stress_max":8}
    assert plan["causal_history_bars_by_provider"]["v3"]["1h"] == 2030
    assert plan["causal_history_bars_by_provider"]["v2_original"]["1h"] == 110
    followup.verify_followup_plan(plan)
    original.atomic_json(parent_dir/"train_results.json",{"changed":True})
    with pytest.raises(ValueError,match="parent research outcomes"):
        followup.verify_followup_plan(plan)


def test_report_labels_v3_and_both_final_cost_variants_once(tmp_path):
    plan = _plan()
    plan.update(experiment="v3-source-followup",source_venue="mexc",execution_resolution="5m",market_trials=1,
                cross_venue_experiment=False,coarse_execution_proxy=True,folds=original.chronological_folds("2026-01-01","2026-01-11","5m"),
                provider_specs={name:asdict(spec) for name,spec in followup.PROVIDERS.items()})
    plan["markets"][0].update(baseline_costs_bps_per_side=[5,2],stress_costs_bps_per_side=[10,5])
    rows = []
    for costs in ("baseline","stress"):
        row = _row(plan["candidates"][0],n=3)
        row.update(phase="final",source_venue="mexc",cost_variant=costs,execution_timeframe="5m",long_trades=2,short_trades=1,
                   tp_hits=2,tp_rate_percent=200/3,win_rate=200/3,funding_total_usdt=0,max_marked_drawdown_percent=1,
                   marked_equity=1010,unfilled_signals=0,small_sample=True)
        rows.append(row)
    _stage(tmp_path,plan,"final",rows)
    followup.write_followup_report(tmp_path,plan)
    report = (tmp_path/"research_report.md").read_text(encoding="utf-8")
    assert "48 основных вариантов" in report and "Provider/context" in report
    assert report.count("v3/latest_leg_daily") == 2
    assert "v3/latest_leg_daily | baseline" in report and "v3/latest_leg_daily | stress" in report
    table = pd.read_csv(tmp_path/"research_summary.csv",encoding="utf-8-sig")
    assert list(table["provider_version"]) == ["experimental_v3"]*2
    assert list(table["context_mode"]) == ["latest_leg_daily"]*2
    assert set(table["costs"]) == {"baseline","stress"}
