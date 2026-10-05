import json
from concurrent.futures import Future
from copy import deepcopy

import pandas as pd
import pytest

from app import research_suite as suite
from app.schemas.setup import TradeSetup


def _row(identifier="chosen", *, category="candidate", n=10, pnl=10, pf=2, unfinished=0, market="core"):
    return {"market": market, "mode": "intraday", "candidate": {"id": identifier, "category": category, "parameters": {}},
            "cost_variant": "baseline", "trades_count": n, "wins": n//2, "net_pnl_usdt": pnl,
            "profit_factor": pf, "return_percent": pnl/10, "unfinished_positions": unfinished}


def _plan(markets=("core",)):
    return {"plan_sha256": "registered", "markets": [{"name": name, "symbols": ["BTC_USDT"]} for name in markets],
            "modes": ["intraday"], "policy": deepcopy(suite.POLICY),
            "candidates": [{"id": "chosen", "category": "candidate", "parameters": {}},
                           {"id": "v1-control", "category": "control", "parameters": None}]}


def _stage(tmp_path, plan, phase, rows):
    value = {"plan_sha256": plan["plan_sha256"], "completed_at_utc": "2026-10-04T00:00:00Z", "results": rows}
    suite.atomic_json(tmp_path/f"{phase}_results.json", value)
    return value


def test_grid_is_finite_unique_and_semantic_ablations_never_main():
    grid = suite.candidate_grid()
    assert len(grid) == len({row["id"] for row in grid}) == 31
    main = [row for row in grid if row["category"] == "candidate"]
    assert len(main) == 24
    assert {row["parameters"]["context_mode"] for row in main} == {"preexisting_target"}
    assert len([row for row in grid if row["category"] == "semantic_ablation"]) == 6
    assert suite.candidate_grid() == grid


def test_folds_are_aligned_nonoverlapping_and_preserve_bar_total():
    folds = suite.chronological_folds("2025-11-08T15:23:06.405Z", "2026-04-07T15:23:06.405Z", "5m")
    assert folds["train"]["start_utc"] == "2025-11-08T15:25:00+00:00"
    assert folds["test"]["end_utc"] == "2026-04-07T15:20:00+00:00"
    assert sum(row["execution_bars"] for row in folds.values()) == 150*288-1
    assert folds["train"]["end_utc"] == folds["validation"]["start_utc"]
    assert folds["validation"]["end_utc"] == folds["test"]["start_utc"]
    assert suite.validate_folds(folds, "5m") == folds
    folds["validation"]["start_utc"] = "2026-02-06T00:01:00Z"
    with pytest.raises(ValueError, match="align"):
        suite.validate_folds(folds, "5m")


def test_external_explicit_calendar_folds_include_leap_day():
    bounds = [("2024-01-01", "2025-01-01"), ("2025-01-01", "2025-07-01"), ("2025-07-01", "2026-01-01")]
    folds = suite.validate_folds({stage: {"start_utc": a, "end_utc": b}
                                 for stage, (a, b) in zip(suite.STAGES, bounds)}, "1m")
    assert [row["execution_bars"]//1440 for row in folds.values()] == [366, 181, 184]


def test_shortlist_requires_floor_positive_flat_and_main_category():
    rows = [_row("too-few", n=9, pf=99), _row("negative", pnl=-1, pf=99),
            _row("ablation", category="semantic_ablation", pf=99), _row("open", unfinished=1, pf=99),
            _row("d", pf=2), _row("c", pf=2), _row("b", pf=3), _row("a", pf=4)]
    assert [row["candidate"]["id"] for row in suite.shortlist(rows)] == ["a", "b", "c"]


def test_validation_only_train_shortlist_never_selects_open_inventory_or_tiny_sample():
    train = [_row("a", pf=4), _row("b", pf=3), _row("c", pf=2), _row("outside", pf=1)]
    val = [_row("outside", n=30, pf=99), _row("a", n=2, pf=20), _row("b", n=3, unfinished=1, pf=10), _row("c", n=3)]
    assert suite.validation_winner(val, train)["candidate"]["id"] == "c"
    assert suite.validation_winner(val[:-1], train) is None


def test_no_validated_candidate_does_not_open_final_jobs():
    plan = _plan()
    selection = {"winners": {"core/intraday": None}}
    assert suite.final_jobs(plan, selection) == []
    verdict = suite.research_verdicts(plan, selection, [])["core/intraday"]
    assert verdict["final_test_opened_for_group"] is False


def test_completed_validation_recovers_selection_write_after_crash(tmp_path, monkeypatch):
    plan = _plan()
    suite.atomic_json(tmp_path/"research_plan.json", plan)
    train = _stage(tmp_path, plan, "train", [_row()])
    val = _stage(tmp_path, plan, "validation", [_row(n=3)])
    monkeypatch.setattr(suite, "verify_plan", lambda value: None)
    monkeypatch.setattr(suite, "write_report", lambda *args: None)
    assert suite.run_phase(tmp_path, "validation") == val
    frozen = json.loads((tmp_path/"frozen_selection.json").read_text(encoding="utf-8"))
    assert frozen["winners"]["core/intraday"]["id"] == "chosen"
    assert suite.freeze_selection(plan, tmp_path, train["results"], val["results"]) == frozen


def test_frozen_selection_rejects_mutated_inputs_and_missing_selection_after_final(tmp_path):
    plan = _plan()
    train = _stage(tmp_path, plan, "train", [_row()])
    val = _stage(tmp_path, plan, "validation", [_row(n=3)])
    frozen = suite.freeze_selection(plan, tmp_path, train["results"], val["results"])
    suite.atomic_json(tmp_path/"final_opened.json", {"plan_sha256": plan["plan_sha256"], "selection_sha256": frozen["selection_sha256"]})
    modified = deepcopy(frozen)
    modified["winners"]["core/intraday"] = None
    with pytest.raises(ValueError, match="modified"):
        suite.validate_selection(plan, tmp_path, modified)
    val["results"][0]["net_pnl_usdt"] += 1
    suite.atomic_json(tmp_path/"validation_results.json", val)
    with pytest.raises(ValueError, match="inputs changed"):
        suite.validate_selection(plan, tmp_path, frozen)
    (tmp_path/"frozen_selection.json").unlink()
    with pytest.raises(ValueError, match="after final"):
        suite.freeze_selection(plan, tmp_path, train["results"], val["results"])


def test_staggered_workers_wait_only_pending_and_gather_all_authoritative_returns(tmp_path, monkeypatch):
    plan = _plan(("first", "second"))
    plan["candidates"] = plan["candidates"][:1]
    suite.atomic_json(tmp_path/"research_plan.json", plan)
    monkeypatch.setattr(suite, "verify_plan", lambda value: None)
    monkeypatch.setattr(suite, "write_report", lambda *args: None)
    future_values, wait_inputs = {}, []

    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, plan, phase, market, mode, candidates, checkpoint):
            future = Future()
            rows = [_row(market=market["name"])]
            future_values[future] = {"market": market["name"], "mode": mode, "results": rows, "signal_cache": {"batch": 1}}
            return future

    def staggered_wait(pending, **kwargs):
        pending = set(pending)
        assert not any(future.done() for future in pending)
        wait_inputs.append(len(pending))
        future = next(iter(pending))
        future.set_result(future_values[future])
        return {future}, pending-{future}

    monkeypatch.setattr(suite, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(suite, "wait", staggered_wait)
    result = suite.run_phase(tmp_path, "train", workers=2)
    assert wait_inputs == [2, 1]
    assert result["completed_at_utc"] and len(result["results"]) == 2
    assert set(result["signal_cache"]) == {"first/intraday", "second/intraday"}


def test_plan_signature_and_engine_hash_detect_drift():
    plan = {"code_sha256": {"config.py": suite.file_hash(suite.Path(suite.__file__).parent/"config.py")}}
    plan["plan_sha256"] = suite.digest(plan)
    suite.verify_plan(plan)
    plan["code_sha256"]["config.py"] = "wrong"
    plan["plan_sha256"] = suite.digest({key: value for key, value in plan.items() if key != "plan_sha256"})
    with pytest.raises(ValueError, match="source changed"):
        suite.verify_plan(plan)


def test_batch_precomputation_uses_only_closed_frames_and_retains_every_variant(monkeypatch):
    from app.strategy import volium_v2
    settings = suite.settings_for("intraday").model_copy(update={"volium_session_enabled": False})
    start = pd.Timestamp("2026-01-01T07:00Z")
    frame_map = {}
    for tf in ("1d", "1h", "5m"):
        index = pd.date_range(end=start+pd.Timedelta(days=1), periods=400, freq=f"{suite.SECONDS[tf]}s")
        if tf == "5m":
            index = pd.date_range(start=start-pd.Timedelta(hours=1), periods=30, freq="5min")
        frame_map[tf] = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.}, index=index)
    candidates = suite.candidate_grid()[:2]
    calls = []
    def batch(*, parameter_sets, **kwargs):
        now = suite.utc(kwargs["now"])
        calls.append(now)
        for tf, frame in kwargs["frames"].items():
            assert (frame.index+pd.Timedelta(seconds=suite.SECONDS[tf]) <= now).all()
        assert kwargs["frames"]["5m"].index[-1]+pd.Timedelta(minutes=5) == now
        return [TradeSetup(id=f"variant-{i}-{now.value}",timestamp=now.to_pydatetime(),symbol=kwargs["symbol"],
                           direction="LONG",setup_type="VOLIUM_INTRADAY",htf_bias="BULLISH",entry=100,
                           stop_loss=99,take_profits=[102],risk_reward=2,confluences=[]) for i in range(len(parameter_sets))]
    monkeypatch.setattr(volium_v2, "analyze_volium_v2_batch_from_df", batch)
    window = {"start_utc": start.isoformat(), "end_utc": (start+pd.Timedelta(minutes=15)).isoformat()}
    tables, metrics = suite.precompute_signals({"BTC_USDT":frame_map}, ["BTC_USDT"], settings, candidates, window)
    assert len(calls) == metrics["batch_structural_calls"] == 3
    assert all(len(table) == 3 for table in tables.values())
    assert tables[candidates[0]["id"]] != tables[candidates[1]["id"]]


def test_registration_verifies_csv_without_invoking_any_strategy_then_refuses_data_drift(tmp_path, monkeypatch):
    cache = tmp_path/"cache"
    cache.mkdir()
    start = pd.Timestamp("2026-01-01T00:00Z")
    manifest = {"status":"complete","server_snapshot_utc":"2026-01-02T00:00Z",
                "frames":{"BTC_USDT":{}},"funding":{"BTC_USDT":{}}}
    for tf in ("1d","1h","5m"):
        index = pd.date_range(start=start-pd.Timedelta(days=5), end=start+pd.Timedelta(hours=2), freq=f"{suite.SECONDS[tf]}s")
        frame = pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.},index=index)
        path = cache/f"BTC_USDT_{tf}.csv"
        frame.to_csv(path)
        manifest["frames"]["BTC_USDT"][tf] = {"sha256":suite.file_hash(path)}
    funding = cache/"BTC_USDT_funding.csv"
    pd.Series([.0001],index=pd.DatetimeIndex([start]),name="funding_rate").to_csv(funding)
    manifest["funding"]["BTC_USDT"] = {"sha256":suite.file_hash(funding),"status":"available"}
    suite.atomic_json(cache/"snapshot.json",manifest)
    def must_not_run(**kwargs):
        raise AssertionError("Preregistration must not inspect strategy outcomes")
    monkeypatch.setattr(suite,"analyze_volium_from_df",must_not_run)
    plan = suite.register_plan(cache,tmp_path/"out",name="synthetic-source-check",markets=[{"name":"core","symbols":["BTC_USDT"]}],
                               modes=["intraday"],resolution="5m",folds=suite.chronological_folds(start,start+pd.Timedelta(minutes=75),"5m"))
    assert plan["coverage"]["BTC_USDT"]["execution_bars"] == 15
    assert plan["market_trials"] == 1 and plan["search_budget"]["train_main"] == 24
    assert plan["markets"][0]["baseline_costs_bps_per_side"] == [5,2]
    suite.verify_plan(plan)
    source = cache/"BTC_USDT_5m.csv"
    source.write_text(source.read_text(encoding="utf-8")+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="data changed"):
        suite.read_dataset(plan)


@pytest.mark.parametrize("last_n,last_pnl,unfinished,positive", [(3, 1, 0, True), (2, 1, 0, False), (3, -1, 0, False), (3, 1, 1, False)])
def test_final_verdict_requires_preregistered_all_fold_conditions(last_n, last_pnl, unfinished, positive):
    plan = _plan()
    candidate = _row()["candidate"]
    selection = {"winners": {"core/intraday": candidate}}
    rows = []
    for phase, n, pnl, open_count in [("train", 10, 1, 0), ("validation", 3, 1, 0), ("final", last_n, last_pnl, unfinished)]:
        row = _row(n=n, pnl=pnl, unfinished=open_count)
        row["phase"] = phase
        rows.append(row)
    verdict = suite.research_verdicts(plan, selection, rows)["core/intraday"]
    assert verdict["positive_historical_candidate"] is positive
    assert verdict["small_sample"] is True
    assert verdict["cost_stress_survived"] is False
