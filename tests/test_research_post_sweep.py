import hashlib
import json
import zipfile
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from app import research_followup as prior
from app import research_post_sweep as research
from app import research_suite as common
from app.schemas.setup import TradeSetup


def _row(candidate, *, n=10, pnl=10, market="core"):
    return {"market":market,"mode":"intraday","candidate":candidate,"cost_variant":"baseline",
            "trades_count":n,"wins":n//2,"net_pnl_usdt":pnl,"return_percent":pnl/10,
            "profit_factor":2,"unfinished_positions":0}


def _plan(markets=("core",)):
    return {"runner":"app.research_post_sweep","plan_sha256":"registered",
            "markets":[{"name":name,"symbols":["BTC_USDT"]} for name in markets],"modes":["intraday"],
            "candidates":research.post_sweep_grid(),"policy":deepcopy(common.POLICY),
            "prior_recorded_research_trials":290,"prior_v1_suite_trials":255}


def _stage(directory, plan, phase, rows):
    value = {"plan_sha256":plan["plan_sha256"],"completed_at_utc":"2026-10-04T00:00:00Z","results":rows}
    common.atomic_json(directory/f"{phase}_results.json",value)
    return value


def _parents(tmp_path, design=None):
    directories = []
    for i,(runner,venue,train_n,val_n) in enumerate([
            ("app.research_suite","mexc",62,2),("app.research_suite","binance_usdm",62,2),
            ("app.research_followup","mexc",102,6),("app.research_followup","binance_usdm",51,3)]):
        directory = tmp_path/f"parent{i}"
        plan = {**(design or {}),"runner":runner,"source_venue":venue,"plan_sha256":f"parent{i}","experiment":f"old{i}"}
        common.atomic_json(directory/"research_plan.json",plan)
        for phase,n in (("train",train_n),("validation",val_n),("final",0)):
            _stage(directory,plan,phase,[{}]*n)
        directories.append(directory)
    return directories


def _archive(tmp_path, entries):
    path = tmp_path/"old_source.zip"
    with zipfile.ZipFile(path,"w") as archive:
        for name,content in entries.items():
            archive.writestr(name,content)
    manifest = {"sha256":common.file_hash(path),"bytes":path.stat().st_size,
                "files":[{"path":name,"sha256":hashlib.sha256(content).hexdigest(),"bytes":len(content)} for name,content in entries.items()]}
    common.atomic_json(path.with_name("old_source_manifest.json"),manifest)
    return path


def test_last_family_is_24_cells_three_controls_and_leaves_prior_registries_unchanged():
    old_v2,old_v3 = common.candidate_grid(),prior.followup_grid()
    grid = research.post_sweep_grid()
    assert len(grid) == len({item["id"] for item in grid}) == 27
    main = [item for item in grid if item["category"] == "candidate"]
    assert len(main) == 24 and {item["provider"] for item in main} == {"v4"}
    assert {item["parameters"]["context_mode"] for item in main} == {"post_sweep_break"}
    assert {item["parameters"]["reaction_min_atr"] for item in main} == {.8,1.2,1.6}
    assert {item["parameters"]["reaction_min_body_ratio"] for item in main} == {.6,.7}
    assert {item["parameters"]["reaction_min_prior_body_ratio"] for item in main} == {1,1.5}
    assert {item["parameters"]["reaction_max_bars"] for item in main} == {2,3}
    assert {item["provider"] for item in grid if item["category"] == "control"} == {"v1","v3_daily","v3_local"}
    assert old_v2 == common.candidate_grid() and old_v3 == prior.followup_grid()
    assert not {item["id"] for item in main} & {item["id"] for item in old_v3}
    assert research.JOINT_POST_SWEEP_BUDGET["train_total"] == 81
    assert research.JOINT_POST_SWEEP_BUDGET["last_context_family_for_this_research"]


def test_draft_never_imports_or_analyzes_new_provider(monkeypatch,capsys):
    monkeypatch.setattr(research,"provider_functions",lambda spec:pytest.fail("Draft must not bind/analyze"))
    research.main(["draft"])
    result = json.loads(capsys.readouterr().out)
    assert result["no_search_executed"] and len(result["candidates"]) == 27


def test_explicit_parameter_and_batch_exports_bind_every_registered_cell_without_analysis():
    for candidate in research.post_sweep_grid():
        factory,batch,diagnostic,single = research.provider_functions(research.PROVIDERS[candidate["provider"]])
        if factory is None:
            assert callable(single)
        else:
            parameters = factory(**candidate["parameters"])
            assert parameters.context_mode == candidate["parameters"]["context_mode"]
            assert parameters.reaction_atr_period == 14
            assert callable(batch) and callable(diagnostic)


def test_all_completed_parents_are_required_and_ledger_counts_prior_545(tmp_path):
    directories = _parents(tmp_path)
    ledger = research.completed_parent_ledger(directories)
    assert sum(stage["trials"] for parent in ledger for stage in parent["phases"].values())+255 == 545
    with pytest.raises(ValueError,match="four"):
        research.completed_parent_ledger(directories[:3])
    (directories[0]/"final_results.json").unlink()
    with pytest.raises(ValueError,match="zero-trial final"):
        research.completed_parent_ledger(directories)


def test_fixed_v3_controls_cannot_rescue_selection_or_open_final():
    plan = _plan()
    controls = [item for item in plan["candidates"] if item["category"] == "control"]
    jobs = research.stage_jobs(plan,"validation",training=[_row(item,pnl=100) for item in controls])
    assert [item["id"] for item in jobs[0][2]] == [item["id"] for item in controls]
    assert research.stage_jobs(plan,"final",selection={"winners":{"core/intraday":None}}) == []
    selected = plan["candidates"][0]
    jobs = research.stage_jobs(plan,"final",selection={"winners":{"core/intraday":selected}})
    assert jobs[0][2] == [selected]+controls


def test_causal_batch_keeps_extended_v4_v3_history_and_original_v1_prefix(monkeypatch):
    settings = common.settings_for("intraday").model_copy(update={"volium_session_enabled":False})
    start = pd.Timestamp("2026-01-01T07:00Z")
    frames = {}
    for tf,periods,end in [("1d",600,start+pd.Timedelta(days=200)),("1h",4000,start+pd.Timedelta(days=10)),
                           ("5m",1000,start+pd.Timedelta(hours=1))]:
        index = pd.date_range(end=end,periods=periods,freq=f"{common.SECONDS[tf]}s")
        frames[tf] = pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.},index=index)
    grid = research.post_sweep_grid()
    candidates = [grid[0],grid[1],*grid[-3:]]
    counts = {}
    def functions(spec):
        def values(n,kwargs):
            now = common.utc(kwargs["now"])
            assert len(kwargs["frames"]["1h"]) == (2030 if spec.h1_history_daily_multiplier else 110)
            for tf,frame in kwargs["frames"].items():
                assert (frame.index+pd.Timedelta(seconds=common.SECONDS[tf]) <= now).all()
            counts[spec.version] = counts.get(spec.version,0)+1
            return [TradeSetup(id=f"{spec.version}-{i}-{now.value}",timestamp=now.to_pydatetime(),symbol=kwargs["symbol"],
                               direction="LONG",setup_type="VOLIUM_INTRADAY",htf_bias="BULLISH",entry=100,
                               stop_loss=99,take_profits=[102],risk_reward=2,confluences=[]) for i in range(n)]
        if spec.parameter_type:
            return (lambda **kwargs:kwargs),(lambda parameter_sets,**kwargs:values(len(parameter_sets),kwargs)),None,None
        return None,None,None,(lambda **kwargs:values(1,kwargs)[0])
    monkeypatch.setattr(research,"provider_functions",functions)
    window = {"start_utc":start.isoformat(),"end_utc":(start+pd.Timedelta(minutes=15)).isoformat()}
    tables,metrics = research.precompute_post_sweep({"BTC_USDT":frames},["BTC_USDT"],settings,candidates,window,
                                                   {name:asdict(spec) for name,spec in research.PROVIDERS.items()})
    assert all(len(table) == 3 for table in tables.values()) and metrics["eligible_time_boundaries"] == 3
    assert counts == {spec.version:3 for spec in research.PROVIDERS.values()}


def test_pair_must_be_registered_and_identical_before_any_trial(tmp_path,monkeypatch):
    plan = _plan()
    plan.update(paired_plan_directory=str(tmp_path),source_venue="mexc",code_sha256={},parent_trials=[],
                joint_post_sweep_budget=research.JOINT_POST_SWEEP_BUDGET)
    monkeypatch.setattr(common,"verify_plan",lambda value:None)
    with pytest.raises(ValueError,match="both"):
        research.verify_registered_pair(plan)
    peer = {**plan,"source_venue":"binance_usdm"}
    common.atomic_json(tmp_path/"research_plan.json",peer)
    research.verify_registered_pair(plan)
    peer["candidates"] = peer["candidates"][:-1]
    common.atomic_json(tmp_path/"research_plan.json",peer)
    with pytest.raises(ValueError,match="differs"):
        research.verify_registered_pair(plan)


def test_archive_verified_per_file_without_extracting_and_rejects_tampering(tmp_path):
    path = _archive(tmp_path,{"safe/file.py":b"original archived bytes"})
    metadata = research.archive_provenance(path,[])
    assert metadata["verified_file_count"] == 1
    manifest_path = path.with_name("old_source_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["sha256"] = "incorrect"
    common.atomic_json(manifest_path,manifest)
    with pytest.raises(ValueError,match="source bytes"):
        research.archive_provenance(path,[])


def test_own_worker_authoritatively_gathers_staggered_jobs_not_completed_futures(tmp_path,monkeypatch):
    plan = _plan(("first","second"))
    plan["candidates"] = plan["candidates"][:1]
    common.atomic_json(tmp_path/"research_plan.json",plan)
    monkeypatch.setattr(research,"verify_post_sweep_plan",lambda value:None)
    monkeypatch.setattr(research,"verify_registered_pair",lambda value:None)
    monkeypatch.setattr(research,"write_post_sweep_report",lambda *args:None)
    values,waits = {},[]
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,plan,phase,market,mode,candidates,checkpoint):
            assert fn is research.run_post_sweep_group and fn is not prior.run_followup_group
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
    monkeypatch.setattr(research,"ProcessPoolExecutor",Pool)
    monkeypatch.setattr(research,"wait",waiter)
    result = research.run_post_sweep_phase(tmp_path,"train",2)
    assert waits == [2,1] and len(result["results"]) == 2 and result["completed_at_utc"]


def test_completed_validation_recovers_selection_and_zero_final_has_no_marker(tmp_path,monkeypatch):
    plan = _plan()
    common.atomic_json(tmp_path/"research_plan.json",plan)
    _stage(tmp_path,plan,"train",[])
    stage = _stage(tmp_path,plan,"validation",[])
    monkeypatch.setattr(research,"verify_post_sweep_plan",lambda value:None)
    monkeypatch.setattr(research,"verify_registered_pair",lambda value:None)
    monkeypatch.setattr(research,"write_post_sweep_report",lambda *args:None)
    assert research.run_post_sweep_phase(tmp_path,"validation") == stage
    frozen = json.loads((tmp_path/"frozen_selection.json").read_text())
    assert frozen["winners"]["core/intraday"] is None
    final = research.run_post_sweep_phase(tmp_path,"final")
    assert final["expected_trials"] == 0 and final["completed_at_utc"]
    assert not (tmp_path/"final_opened.json").exists()


def test_registration_records_unchanged_parent_design_and_545_prior_trials_without_analysis(tmp_path,monkeypatch):
    cache = tmp_path/"data"
    cache.mkdir()
    start = pd.Timestamp("2026-01-01T00:00Z")
    manifest = {"status":"complete","server_snapshot_utc":"2026-01-02T00:00Z",
                "frames":{"BTC_USDT":{}},"funding":{"BTC_USDT":{}}}
    for tf in ("1d","1h","5m"):
        index = pd.date_range(start=start-pd.Timedelta(days=5),end=start+pd.Timedelta(hours=2),freq=f"{common.SECONDS[tf]}s")
        frame = pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.},index=index)
        path = cache/f"BTC_USDT_{tf}.csv"
        frame.to_csv(path)
        manifest["frames"]["BTC_USDT"][tf] = {"sha256":common.file_hash(path)}
    funding = cache/"BTC_USDT_funding.csv"
    pd.Series([.0001],index=pd.DatetimeIndex([start]),name="funding_rate").to_csv(funding)
    manifest["funding"]["BTC_USDT"] = {"sha256":common.file_hash(funding),"status":"available"}
    common.atomic_json(cache/"snapshot.json",manifest)
    markets = [{"name":"core","symbols":["BTC_USDT"],"baseline_costs_bps_per_side":[5,2],"stress_costs_bps_per_side":[10,5]}]
    folds = common.chronological_folds(start,start+pd.Timedelta(minutes=75),"5m")
    design = {"data_manifest_sha256":common.file_hash(cache/"snapshot.json"),"markets":markets,"modes":["intraday"],
              "execution_resolution":"5m","folds":folds,"position_limits":{},
              "settings_by_mode":{"intraday":common.settings_for("intraday").model_dump(mode="json")}}
    parents = _parents(tmp_path,design)
    root = Path(research.__file__).parent
    filenames = [*common.CODE_FILES,"research_followup.py","strategy/volium_v3.py"]
    archive = _archive(tmp_path,{"mexc-volium/app/"+filename:(root/filename).read_bytes() for filename in filenames})
    review = tmp_path/"source.md"
    review.write_text("Reviewed source hypothesis",encoding="utf-8")
    def no_outcomes(spec):
        def forbidden(**kwargs):
            raise AssertionError("Registration must not calculate outcomes")
        return ((lambda **kwargs:kwargs),forbidden,forbidden,None) if spec.parameter_type else (None,None,None,forbidden)
    monkeypatch.setattr(research,"provider_functions",no_outcomes)
    output = tmp_path/"v4"
    plan = research.register_post_sweep(cache,output,name="synthetic-v4",markets=deepcopy(markets),modes=["intraday"],
        resolution="5m",folds=folds,parent_directories=parents,source_review=review,source_archive=archive,
        peer_out_dir=tmp_path/"other_venue")
    assert plan["prior_stage_trials_total"] == 545 and len(plan["parent_trials"]) == 4
    assert plan["search_budget"] == {"train_main":24,"train_controls":3,"validation_max":6,"final_baseline_and_stress_max":8}
    assert plan["causal_history_bars_by_provider"]["v3_local"]["1h"] == 2030
    assert plan["causal_history_bars_by_provider"]["v1"]["1h"] == 110
    assert plan["post_a_break_may_be_countertrend"] and plan["not_full_author_replica"]
    research.verify_post_sweep_plan(plan)
    common.atomic_json(parents[0]/"train_results.json",{"changed":True})
    with pytest.raises(ValueError,match="parent research outcomes"):
        research.verify_post_sweep_plan(plan)


def test_reports_label_v4_and_both_final_costs_with_reuse_and_context_weakness(tmp_path):
    plan = _plan()
    plan.update(experiment="v4-post-sweep",source_venue="mexc",execution_resolution="5m",market_trials=1,
                cross_venue_experiment=False,coarse_execution_proxy=True,folds=common.chronological_folds("2026-01-01","2026-01-11","5m"),
                provider_specs={name:asdict(spec) for name,spec in research.PROVIDERS.items()})
    plan["markets"][0].update(baseline_costs_bps_per_side=[5,2],stress_costs_bps_per_side=[10,5])
    rows = []
    for costs in ("baseline","stress"):
        row = _row(plan["candidates"][0],n=3)
        row.update(phase="final",source_venue="mexc",cost_variant=costs,execution_timeframe="5m",long_trades=2,short_trades=1,
                   tp_hits=2,tp_rate_percent=200/3,win_rate=200/3,funding_total_usdt=0,max_marked_drawdown_percent=1,
                   marked_equity=1010,unfilled_signals=0,small_sample=True)
        rows.append(row)
    _stage(tmp_path,plan,"final",rows)
    research.write_post_sweep_report(tmp_path,plan)
    report = (tmp_path/"research_report.md").read_text(encoding="utf-8")
    assert "24 post_sweep_break" in report and "545 stage trials" in report
    assert "против глобального тренда" in report and "Validation не объявляется нетронутым" in report
    assert report.count("v4/post_sweep_break") == 2
    table = pd.read_csv(tmp_path/"research_summary.csv",encoding="utf-8-sig")
    assert list(table["provider_version"]) == ["experimental_v4"]*2
    assert set(table["costs"]) == {"baseline","stress"}
