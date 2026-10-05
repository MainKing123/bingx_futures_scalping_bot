import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from app import research_data as data


def candles(start, count, tf="1m"):
    return pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 10.0},
        index=pd.date_range(start, periods=count, freq=pd.Timedelta(seconds=data.SECONDS[tf])))


def original_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "EXTEND_DAYS", {"1m": 1, "5m": 1})
    monkeypatch.setattr(data, "CHUNK_DAYS", {"1m": 1, "5m": 1})
    source = tmp_path / "suite"
    source.mkdir()
    manifest = {"snapshot_ms": data.FROZEN_MS, "frames": {"BTC_USDT": {}}, "funding": {}}
    for tf in ("1m", "5m", "1h"):
        step = pd.Timedelta(seconds=data.SECONDS[tf])
        end = pd.Timestamp(data.FROZEN_MS, unit="ms", tz="UTC").floor(step)
        frame = candles(end - step * 10, 10, tf)
        path = source / f"BTC_USDT_{tf}.csv"
        frame.to_csv(path, index_label="timestamp")
        manifest["frames"]["BTC_USDT"][tf] = {**data.validate_frame(frame, tf, data.FROZEN_MS),
            "sha256": data.sha256_file(path), "requested_bars": 10}
    funding = pd.DataFrame({"funding_rate": 0.0001},
        index=pd.date_range(pd.Timestamp(data.FROZEN_MS, unit="ms", tz="UTC").floor("8h") - pd.Timedelta(days=400), periods=1201, freq="8h"))
    path = source / "BTC_USDT_funding.csv"
    funding.to_csv(path, index_label="settlement_utc")
    manifest["funding"]["BTC_USDT"] = {"status": "available", "sha256": data.sha256_file(path), "settlements": len(funding)}
    (source / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    original_hashes = {path.name: data.sha256_file(path) for path in source.iterdir()}

    async def fetch(symbol, tf, *, limit, start_time, end_time):
        step = data.SECONDS[tf] * 1000
        assert symbol == "BTC_USDT" and 0 < limit < 100_000
        assert end_time <= data.FROZEN_MS
        return candles(pd.Timestamp(start_time, unit="ms", tz="UTC"), (end_time - start_time) // step, tf).tail(limit)

    client = SimpleNamespace(get_klines=AsyncMock(side_effect=fetch))
    return source, original_hashes, client


def test_public_extension_is_gap_free_frozen_resumable_and_preserves_original(tmp_path, monkeypatch):
    source, hashes, client = original_snapshot(tmp_path, monkeypatch)
    cache = tmp_path / "research"
    manifest = asyncio.run(data.fetch_research_snapshot(source, cache, symbols=("BTC_USDT",), client=client))
    assert manifest["status"] == "complete"
    assert manifest["snapshot_ms"] == data.FROZEN_MS
    for tf in ("1m", "5m"):
        info = manifest["frames"]["BTC_USDT"][tf]
        assert info["bars"] == 86400 // data.SECONDS[tf] + data.CONTEXT_BARS
        assert info["missing_intervals"] == 0
        assert len(info["timestamp_stability_probes"]) == 3
        assert all(probe["matches_cached_data"] for probe in info["timestamp_stability_probes"])
        assert pd.Timestamp(info["last_close_utc"]) <= pd.Timestamp(data.FROZEN_MS, unit="ms", tz="UTC")
        assert data.sha256_file(cache / f"BTC_USDT_{tf}.csv") == info["sha256"]
    assert data.sha256_file(cache / "BTC_USDT_1h.csv") == hashes["BTC_USDT_1h.csv"]
    assert data.sha256_file(cache / "BTC_USDT_funding.csv") == hashes["BTC_USDT_funding.csv"]
    assert {path.name: data.sha256_file(path) for path in source.iterdir()} == hashes
    assert all(call.kwargs["limit"] < 100_000 for call in client.get_klines.await_args_list)
    client.get_klines.reset_mock()
    resumed = asyncio.run(data.fetch_research_snapshot(source, cache, symbols=("BTC_USDT",), client=client))
    assert resumed["status"] == "complete"
    client.get_klines.assert_not_awaited()


def test_empty_historical_api_response_fails_closed_with_incomplete_manifest(tmp_path, monkeypatch):
    source, hashes, client = original_snapshot(tmp_path, monkeypatch)
    client.get_klines.side_effect = None
    client.get_klines.return_value = candles(pd.Timestamp(data.FROZEN_MS, unit="ms", tz="UTC"), 0)
    cache = tmp_path / "research"
    with pytest.raises(data.ResearchDataError, match="Empty"):
        asyncio.run(data.fetch_research_snapshot(source, cache, symbols=("BTC_USDT",), client=client))
    manifest = json.loads((cache / "snapshot.json").read_text())
    assert manifest["status"] == "incomplete"
    assert not (cache / "BTC_USDT_1m.csv").exists()
    assert {path.name: data.sha256_file(path) for path in source.iterdir()} == hashes


def test_changed_original_and_research_hashes_are_rejected(tmp_path, monkeypatch):
    source, _, client = original_snapshot(tmp_path, monkeypatch)
    cache = tmp_path / "research"
    asyncio.run(data.fetch_research_snapshot(source, cache, symbols=("BTC_USDT",), client=client))
    target = cache / "BTC_USDT_1m.csv"
    target.write_text(target.read_text() + "\n", encoding="utf-8")
    with pytest.raises(data.ResearchDataError, match="hash mismatch"):
        asyncio.run(data.fetch_research_snapshot(source, cache, symbols=("BTC_USDT",), client=client))
    parent = source / "snapshot.json"
    parent.write_text(parent.read_text() + "\n", encoding="utf-8")
    with pytest.raises(data.ResearchDataError, match="provenance changed"):
        asyncio.run(data.fetch_research_snapshot(source, cache, symbols=("BTC_USDT",), client=client))


@pytest.mark.parametrize("corruption", ["gap", "duplicate", "future", "range", "nan"])
def test_missing_or_invalid_causal_data_is_rejected(corruption):
    cutoff = pd.Timestamp(data.FROZEN_MS, unit="ms", tz="UTC").floor("min")
    frame = candles(cutoff - pd.Timedelta(minutes=3), 3)
    if corruption == "gap":
        frame = frame.drop(frame.index[1])
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.tail(1)])
    elif corruption == "future":
        frame.index = frame.index + pd.Timedelta(minutes=1)
    elif corruption == "range":
        frame.iloc[0, frame.columns.get_loc("high")] = 98
    else:
        frame.iloc[0, frame.columns.get_loc("volume")] = float("nan")
    with pytest.raises(data.ResearchDataError):
        data.validate_frame(frame, "1m", data.FROZEN_MS)


def test_research_periods_are_chronological_and_seen_context_is_disclosed():
    periods = data.chronological_periods(data.FROZEN_MS)
    research = periods["new_intraday_and_scalp_outcomes"]
    assert research["not_pristine_all_market_context"]
    train, validation, final = (research[name] for name in ("train", "validation", "final_test"))
    assert train["end_utc_exclusive"] == validation["start_utc"]
    assert validation["end_utc_exclusive"] == final["start_utc"]
    assert final["end_utc_exclusive"] == periods["previously_inspected_intraday"]["start_utc"]
    assert research["split_days"] == [111, 37, 37]
    assert all(pd.Timestamp(part["start_utc"]).second == 0 for part in (train, validation, final))


def test_m5_only_mode_keeps_m1_as_legacy_control_and_uses_own_specification(tmp_path, monkeypatch):
    source, hashes, client = original_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(data, "CHUNK_DAYS", {"5m": 60})
    cache = tmp_path / "m5_research"
    manifest = asyncio.run(data.fetch_research_snapshot(source, cache, symbols=("BTC_USDT",), client=client,
        mode="m5_only", history_days=181))
    assert manifest["execution_resolution"] == "5m"
    assert manifest["old_period_source_scalp_M1_unavailable"]
    assert manifest["frames"]["BTC_USDT"]["1m"]["role"] == "legacy_fine_resolution_control_only"
    assert data.sha256_file(cache / "BTC_USDT_1m.csv") == hashes["BTC_USDT_1m.csv"]
    assert all(call.args[1] == "5m" for call in client.get_klines.await_args_list)
    assert manifest["frames"]["BTC_USDT"]["5m"]["bars"] == 181 * 288 + 130
    with pytest.raises(data.ResearchDataError, match="specification changed"):
        asyncio.run(data.fetch_research_snapshot(source, cache, symbols=("BTC_USDT",), client=client,
            mode="m5_only", history_days=330))


@pytest.mark.parametrize("native_has_gap", [False, True])
def test_native_fx_requires_complete_data_and_funding_without_imputing_gaps(tmp_path, monkeypatch, native_has_gap):
    source, _, client = original_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(data, "CHUNK_DAYS", {"5m": 60})
    original_fetch = client.get_klines.side_effect

    async def fetch(symbol, tf, *, limit, end_time, start_time=None):
        if symbol == "BTC_USDT":
            return await original_fetch(symbol, tf, limit=limit, start_time=start_time, end_time=end_time)
        step_ms = data.SECONDS[tf] * 1000
        end = end_time // step_ms * step_ms
        start = start_time if start_time is not None else end - limit * step_ms
        frame = candles(pd.Timestamp(start, unit="ms", tz="UTC"), (end - start) // step_ms, tf).tail(limit)
        if native_has_gap and tf == "5m" and limit > 1:
            return frame.drop(frame.index[len(frame) // 2])
        return frame

    async def funding(symbol, since_ms):
        first = pd.Timestamp(since_ms, unit="ms", tz="UTC").ceil("4h")
        last = pd.Timestamp(data.FROZEN_MS, unit="ms", tz="UTC").floor("4h")
        return pd.Series(0.0001, index=pd.date_range(first, last, freq="4h"), name="funding_rate")

    client.get_klines.side_effect = fetch
    client.get_contract = AsyncMock(return_value={"symbol": "EUR_USDT", "apiAllowed": True, "state": 0,
        "futureType": 1, "quoteCoin": "USDT", "settleCoin": "USDT", "baseCoin": "EUR", "contractSize": 1})
    client.get_funding_history = AsyncMock(side_effect=funding)
    cache = tmp_path / "research"
    manifest = asyncio.run(data.fetch_research_snapshot(source, cache, symbols=("BTC_USDT", "EUR_USDT"),
        client=client, mode="m5_only", history_days=181))
    assert manifest["status"] == "complete"
    assert manifest["requested_symbols"] == ["BTC_USDT", "EUR_USDT"]
    if native_has_gap:
        assert manifest["symbols"] == ["BTC_USDT"]
        assert "EUR_USDT" in manifest["excluded_symbols"]
        assert not (cache / "EUR_USDT_5m.csv").exists()
        client.get_funding_history.assert_not_awaited()
    else:
        assert manifest["symbols"] == ["BTC_USDT", "EUR_USDT"]
        assert manifest["frames"]["EUR_USDT"]["5m"]["missing_intervals"] == 0
        assert manifest["funding"]["EUR_USDT"]["status"] == "available"
        assert len(manifest["frames"]["EUR_USDT"]["5m"]["timestamp_stability_probes"]) == 3
