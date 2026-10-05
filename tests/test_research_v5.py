"""Finite precommit/causal cache/completion tests; no real historical outcome."""
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace
import time

import pandas as pd
import pytest

from app import research_v5 as study
from app.schemas.setup import TradeSetup
from app.runtime_settings import RuntimeSettings
from app.replay import SECONDS


def row(model="v5-strict-cost",phase="train",cost="baseline",mode="intraday",n=30,pnl=10):
    candidate=next(item for item in study.CANDIDATES if item["id"]==model)
    return {"candidate":candidate,"phase":phase,"cost_variant":cost,"market":"crypto_core","mode":mode,
            "trades_count":n,"net_pnl_usdt":pnl,"unfinished_positions":0,"unfinished_limits":0}


def complete_rows(model="v5-strict-cost",n=30):
    return [row(model,phase,cost,n=n) for phase,cost in (("train","baseline"),("validation","baseline"),
        ("validation","stress"),("final","baseline"),("final","stress"))]


def test_joint_budget_is_exactly_57_fixed_trials_no_grid_or_rank():
    mexc={"modes":["intraday"],"robustness_data":{}}
    # A registered robustness object is nonempty by construction.
    mexc["robustness_data"]={"window":"known"}
    binance={"modes":["intraday","scalp"],"robustness_data":None}
    counts={phase:len(study.expected_identities(mexc,phase))+len(study.expected_identities(binance,phase))
            for phase in ("train","validation","final","robustness")}
    assert counts=={"train":9,"validation":18,"final":18,"robustness":12}
    assert sum(counts.values())==57
    assert len(study.CANDIDATES)==3 and study.POLICY["ranking"] is False
    assert study.ROBUST_FIVE==["BTC_USDT","ETH_USDT","ZEC_USDT","SOL_USDT","DOGE_USDT"]


def test_zero_training_does_not_remove_fixed_final_models():
    plan={"modes":["intraday","scalp"],"robustness_data":None}
    assert len(study.expected_identities(plan,"final"))==12
    assert {identity[2] for identity in study.expected_identities(plan,"final")}=={item["id"] for item in study.CANDIDATES}


def test_mexc_full_330d_folds_are_fixed_aligned_and_cover_all_days():
    folds=study.approved_folds("mexc",[])
    assert folds["train"]["execution_bars"]==43199
    assert folds["validation"]["execution_bars"]==25920
    assert folds["test"]["execution_bars"]==25920
    assert folds["test"]["end_utc"]=="2026-10-04T15:20:00+00:00"
    assert folds["train"]["end_utc"]==folds["validation"]["start_utc"]


def test_runtime_and_legacy_settings_do_not_load_credentials_or_change_frozen_leverage():
    repaired=study.runtime_settings("intraday")
    legacy=study.common.settings_for("intraday")
    assert repaired.default_leverage==50 and legacy.default_leverage==3
    assert repaired.auto_execution is False and repaired.mexc_api_key=="" and repaired.mexc_api_secret==""
    assert repaired.paper_fee_bps==5 and repaired.paper_slippage_bps==2
    assert study.runtime_settings("scalp",stress=True).paper_slippage_bps==5


def test_authoritative_collection_rejects_missing_and_duplicate_trials():
    expected={("crypto_core","intraday","v5-strict-cost","baseline")}
    assert len(study.collect_complete([{"results":[row()]}],expected))==1
    with pytest.raises(ValueError,match="Duplicate"):
        study.collect_complete([{"results":[row(),row()]}],expected)
    with pytest.raises(ValueError,match="registered fixed budget"):
        study.collect_complete([{"results":[]}],expected)


def test_eligibility_requires_positive_folds_and_15_each_not_sum():
    values=complete_rows()
    assert study.eligibility(values,"intraday","v5-strict-cost")["eligible"] is True
    values[1]["trades_count"]=14
    outcome=study.eligibility(values,"intraday","v5-strict-cost")
    assert outcome["eligible"] is False
    assert "validation_fewer_than_15_closed" in outcome["reasons"]
    assert "validation" in outcome["small_samples"]
    values=complete_rows();values[4]["net_pnl_usdt"]=-.01
    assert "final_stress_nonpositive" in study.eligibility(values,"intraday","v5-strict-cost")["reasons"]


def test_open_inventory_and_legacy_cannot_be_claimed_eligible():
    values=complete_rows();values[3]["unfinished_limits"]=1
    assert "final_baseline_unfinished_inventory" in study.eligibility(values,"intraday","v5-strict-cost")["reasons"]
    legacy=study.eligibility(complete_rows("v1-legacy3x"),"intraday","v1-legacy3x")
    assert legacy["eligible"] is False and "legacy_control_not_candidate" in legacy["reasons"]


def test_robustness_cannot_raise_eligibility_sample_count():
    values=complete_rows(n=1)
    bonus=row(n=1000);bonus.update(market="core_plus_top3",phase="robustness")
    result=study.eligibility(values+[bonus],"intraday","v5-strict-cost")
    assert result["baseline_total_closed"]==3 and result["eligible"] is False


def test_precompute_never_passes_future_context_and_legacy_keeps_old_history(monkeypatch):
    config=study.runtime_settings("intraday");config.volium_session_enabled=False
    frames={"BTC_USDT":{}}
    for tf in ("1d","1h","5m"):
        idx=pd.date_range(end="2026-01-02T02:00Z",periods=350,freq=f"{SECONDS[tf]}s")
        frames["BTC_USDT"][tf]=pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.},index=idx)
    window={"start_utc":"2026-01-01T00:00Z","end_utc":"2026-01-01T00:10Z"}
    calls=[]
    def check(kwargs):
        for tf,frame in kwargs["frames"].items():
            assert (frame.index+pd.Timedelta(seconds=SECONDS[tf])<=kwargs["now"]).all()
    def batch(parameter_sets,**kwargs):
        check(kwargs);calls.append(kwargs)
        return [None,None]
    def legacy(**kwargs):
        check(kwargs)
        assert all(len(frame)<=110 for frame in kwargs["frames"].values())
        return None
    module=SimpleNamespace(V5Parameters=lambda **kwargs:kwargs,analyze_volium_v5_batch_from_df=batch)
    monkeypatch.setattr(study.importlib,"import_module",lambda name:module)
    monkeypatch.setattr(study,"analyze_volium_from_df",legacy)
    tables,metrics=study.precompute(frames,["BTC_USDT"],config,window,study.history_requirements("intraday"))
    assert metrics["session_confirmation_checks"]==2 and len(calls)==2
    assert all(not table for table in tables.values())


def test_data_registration_coverage_checks_timestamps_only_and_aborts_gap(tmp_path,monkeypatch):
    manifest={"status":"complete","server_snapshot_utc":"2026-01-02T00:00Z"}
    (tmp_path/"snapshot.json").write_text(json.dumps(manifest),encoding="utf-8")
    folds=study.common.chronological_folds("2026-01-01T00:00Z","2026-01-01T00:10Z","1m")
    frame=pd.DataFrame(index=pd.date_range("2026-01-01T00:00Z",periods=10,freq="min"))
    frame=frame.drop(frame.index[3])
    monkeypatch.setattr(study.common,"source_paths",lambda *args:{})
    monkeypatch.setattr(study.common,"read_dataset",lambda plan:({"BTC_USDT":{"1m":frame}},{}))
    with pytest.raises(ValueError,match="Missing 1 execution bars"):
        study.data_spec(tmp_path,["BTC_USDT"],["scalp"],"1m",folds)


def test_staggered_workers_complete_all_authoritative_results_and_idempotent_recovery(tmp_path,monkeypatch):
    plan={"plan_sha256":"test","runner":"app.research_v5","modes":["intraday","scalp"],"robustness_data":None}
    study.common.atomic_json(tmp_path/"research_plan.json",plan)
    monkeypatch.setattr(study,"verify",lambda value:None)
    monkeypatch.setattr(study,"write_report",lambda *args:None)
    monkeypatch.setattr(study,"ProcessPoolExecutor",ThreadPoolExecutor)
    def worker(plan,phase,market,mode,stress,checkpoint):
        time.sleep(.01 if mode=="intraday" else .08)
        return {"market":market,"mode":mode,"signal_cache":{},"results":[row(model=item["id"],mode=mode) for item in study.CANDIDATES]}
    monkeypatch.setattr(study,"run_group",worker)
    result=study.run_phase(tmp_path,"train",2)
    assert result["completed_at_utc"] and len(result["results"])==6
    assert len({(value["mode"],value["candidate"]["id"]) for value in result["results"]})==6
    monkeypatch.setattr(study,"run_group",lambda *args:pytest.fail("Completed stage should not repeat outcomes"))
    assert study.run_phase(tmp_path,"train",2)==result


def test_final_cannot_open_before_full_earlier_phases(tmp_path,monkeypatch):
    plan={"plan_sha256":"test","runner":"app.research_v5","modes":["intraday"],"robustness_data":None}
    study.common.atomic_json(tmp_path/"research_plan.json",plan)
    monkeypatch.setattr(study,"verify",lambda value:None)
    with pytest.raises(ValueError,match="earlier stages"):
        study.run_phase(tmp_path,"final")
    assert not (tmp_path/"final_opened.json").exists()


def test_earlier_incomplete_stages_cannot_reopen_after_final_inspection(tmp_path,monkeypatch):
    plan={"plan_sha256":"test","runner":"app.research_v5","modes":["intraday"],"robustness_data":None}
    study.common.atomic_json(tmp_path/"research_plan.json",plan)
    study.common.atomic_json(tmp_path/"final_opened.json",{"plan_sha256":"test"})
    monkeypatch.setattr(study,"verify",lambda value:None)
    with pytest.raises(ValueError,match="reopen earlier"):
        study.run_phase(tmp_path,"train")


def test_draft_is_network_free_and_does_not_import_provider(monkeypatch,capsys):
    monkeypatch.setattr(study.importlib,"import_module",lambda name:pytest.fail("Draft must not analyze provider"))
    study.main(["draft"])
    payload=json.loads(capsys.readouterr().out)
    assert payload["no_outcomes_computed"] is True
    assert payload["policy"]["finite_joint_budget"]["total"]==57
