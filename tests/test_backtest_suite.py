import asyncio
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import backtest_suite as suite


def _payload(symbol, scenarios, hits=0):
    return {"symbol": symbol,
            "results": [{"symbol": symbol, "scenario": asdict(scenario)} for scenario in scenarios],
            "signal_cache": {"hits": hits, "misses": 0, "broad_rejections": 0, "broad_evaluations": 0}}


def test_final_gather_validates_and_orders_worker_cases():
    scenarios = [suite.Scenario("first", "intraday", 180), suite.Scenario("second", "scalp", 30)]
    payloads = [_payload("ETH_USDT", scenarios, hits=3), _payload("ZEC_USDT", scenarios, hits=5)]
    # Identical repeated records are harmless; conflicting records must fail.
    payloads[0]["results"].append(deepcopy(payloads[0]["results"][0]))
    results, metrics = suite.merge_completed_payloads(payloads, scenarios, ["ZEC_USDT", "ETH_USDT"])
    assert [(r["scenario"]["name"], r["symbol"]) for r in results] == [
        ("first", "ZEC_USDT"), ("first", "ETH_USDT"), ("second", "ZEC_USDT"), ("second", "ETH_USDT")]
    assert metrics["hits"] == 8


@pytest.mark.parametrize("failure", ["missing", "unexpected", "conflicting"])
def test_final_gather_rejects_incomplete_or_conflicting_worker_results(failure):
    scenarios = [suite.Scenario("first", "intraday", 180)]
    payload = _payload("ZEC_USDT", scenarios)
    if failure == "missing":
        payload["results"] = []
    elif failure == "unexpected":
        payload["results"][0]["symbol"] = "BTC_USDT"
    else:
        changed = deepcopy(payload["results"][0])
        changed["net_pnl_usdt"] = 5
        payload["results"].append(changed)
    with pytest.raises(RuntimeError):
        suite.merge_completed_payloads([payload], scenarios, ["ZEC_USDT"])


def test_worker_completion_between_checkpoint_read_and_done_uses_returned_payload(tmp_path, monkeypatch):
    scenarios = [suite.Scenario("first", "intraday", 180), suite.Scenario("second", "scalp", 30)]
    symbols = list(suite.SYMBOLS)
    frames = {symbol: {} for symbol in symbols}
    snapshot = {"server_snapshot_utc": "2026-10-04T15:23:06.405000+00:00"}
    monkeypatch.setattr(suite, "build_scenarios", lambda *args: scenarios)
    monkeypatch.setattr(suite, "fetch_snapshot", AsyncMock(return_value=(frames, snapshot)))
    monkeypatch.setattr(suite, "funding_snapshot", AsyncMock(return_value={symbol: None for symbol in symbols}))
    written = []
    monkeypatch.setattr(suite, "write_outputs", lambda out, report: written.append(deepcopy(report)))

    class FinishedAfterCheckpointRead:
        def __init__(self, payload):
            self.payload = payload

        def done(self):
            # The worker completed without any progress file having been read.
            return True

        def result(self):
            return self.payload

    class FakePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, fn, symbol, frames, job_scenarios, *args):
            return FinishedAfterCheckpointRead(_payload(symbol, job_scenarios))

    monkeypatch.setattr(suite, "ProcessPoolExecutor", FakePool)
    args = SimpleNamespace(universe_json=None, symbol=symbols, skip_weekly=False,
                           skip_fine=False, scenario=None, dry_run=False, cache=str(tmp_path / "cache"),
                           out_dir=str(tmp_path / "out"), refresh_cache=False, cache_only=True,
                           fetch_only=False, workers=3)
    asyncio.run(suite.run(args))
    assert written[0]["results"] == []
    assert "completed_at_utc" not in written[0]
    final = written[-1]
    assert len(final["results"]) == len(scenarios) * len(symbols) == 10
    assert final["completed_at_utc"]
    assert set((r["scenario"]["name"], r["symbol"]) for r in final["results"]) == {
        (scenario.name, symbol) for scenario in scenarios for symbol in symbols}
