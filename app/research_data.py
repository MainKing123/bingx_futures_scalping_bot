"""Extend the immutable public suite snapshot for preregistered research outcomes.

The original cache is read only. Public requests use blank credentials, bounded
chunks and one shared client's rate limiter; this module never executes a replay.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Settings
from app.exchange.client import MEXCClient

SYMBOLS = ("ZEC_USDT", "SOL_USDT", "DOGE_USDT", "XRP_USDT", "ETH_USDT")
FROZEN_MS = 1791127386405
SECONDS = {"1m": 60, "5m": 300, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
EXTEND_DAYS = {"1m": 365, "5m": 365}
CHUNK_DAYS = {"1m": 30, "5m": 60}
CONTEXT_BARS = 130
PUBLIC_SOURCE = "https://www.mexc.com/api-docs/futures/market-endpoints/get-candlestick-data"


class ResearchDataError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path, frame, index_label="timestamp"):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index_label=index_label)
    temporary.replace(path)


def load_frame(path):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def validate_frame(frame, timeframe, cutoff_ms, *, start_ms=None, boundary_ms=None):
    """Reject any missing, duplicated, invalid, unordered or unclosed source bar."""
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ResearchDataError("Empty or non-UTC public candle data")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ResearchDataError("Duplicate or unordered public candle data")
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(frame.columns):
        raise ResearchDataError("Missing public OHLCV columns")
    values = frame[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ResearchDataError("Invalid public OHLCV values")
    if frame.high.lt(frame[["open", "low", "close"]].max(axis=1)).any() or frame.low.gt(frame[["open", "high", "close"]].min(axis=1)).any():
        raise ResearchDataError("Invalid public OHLC range")
    seconds = SECONDS[timeframe]
    step = pd.Timedelta(seconds=seconds)
    if not frame.index.to_series().diff().dropna().eq(step).all():
        raise ResearchDataError("Missing public candle intervals")
    cutoff = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    if (frame.index + step > cutoff).any():
        raise ResearchDataError("Public candle closes after the frozen cutoff")
    if start_ms is not None and boundary_ms is not None:
        count = (boundary_ms - start_ms) // (seconds * 1000)
        expected = pd.date_range(pd.Timestamp(start_ms, unit="ms", tz="UTC"), periods=count, freq=step)
        if not frame.index.equals(expected):
            raise ResearchDataError("Public candle chunk has incomplete causal coverage")
    return {"bars": len(frame), "first_open_utc": frame.index[0].isoformat(),
        "last_open_utc": frame.index[-1].isoformat(),
        "last_close_utc": (frame.index[-1] + step).isoformat(), "missing_intervals": 0,
        "source": PUBLIC_SOURCE}


def chronological_periods(cutoff_ms, history_days=365):
    cutoff = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    start = (cutoff - pd.Timedelta(days=history_days)).ceil("min")
    end = (cutoff - pd.Timedelta(days=180)).ceil("min")
    new_days = history_days - 180
    train_days = int(new_days * 0.6)
    validation_days = int(new_days * 0.2)
    test_days = new_days - train_days - validation_days
    validation = start + pd.Timedelta(days=train_days)
    final_test = validation + pd.Timedelta(days=validation_days)
    def block(left, right):
        return {"start_utc": left.isoformat(), "end_utc_exclusive": right.isoformat()}
    return {"new_intraday_and_scalp_outcomes": {
        "label": "newly evaluated outcomes; historical H1/D context and aggregate swing outcomes were previously inspected",
        "train": block(start, validation), "validation": block(validation, final_test),
        "final_test": block(final_test, end), "split_days": [train_days, validation_days, test_days],
        "not_pristine_all_market_context": True},
        "previously_inspected_intraday": block(end, cutoff),
        "previously_inspected_scalp_legacy_regression": block((cutoff - pd.Timedelta(days=30)).ceil("min"), cutoff),
        "previously_inspected_swing_context": block(start, cutoff)}


def verify_file(path, recorded):
    if not Path(path).exists() or sha256_file(path) != recorded.get("sha256"):
        raise ResearchDataError(f"Verified source hash mismatch: {Path(path).name}")


async def fetch_research_snapshot(original_cache=".cache/suite", cache=".cache/research", *,
    symbols=SYMBOLS, frozen_ms=FROZEN_MS, client=None, concurrency=3,
    mode="full", history_days=365):
    if mode not in {"full", "m5_only"} or not 181 <= history_days <= 365:
        raise ResearchDataError("Invalid preregistered data mode or history length")
    extension_days = EXTEND_DAYS if mode == "full" else {"5m": history_days}
    original_cache, cache = Path(original_cache), Path(cache)
    if original_cache.resolve() == cache.resolve():
        raise ResearchDataError("Research cache must be separate from the original snapshot")
    original_path = original_cache / "snapshot.json"
    original_hash = sha256_file(original_path)
    parent = json.loads(original_path.read_text(encoding="utf-8"))
    if parent["snapshot_ms"] != frozen_ms:
        raise ResearchDataError("Original snapshot uses a different frozen cutoff")
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "chunks").mkdir(exist_ok=True)
    path = cache / "snapshot.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["snapshot_ms"] != frozen_ms or manifest["parent_snapshot_sha256"] != original_hash:
            raise ResearchDataError("Research snapshot provenance changed; use a different research cache")
        if manifest.get("mode", "full") != mode or manifest.get("requested_history_days", 365) != history_days:
            raise ResearchDataError("Research data specification changed; use a different research cache")
    else:
        manifest = {"source": "MEXC public futures research extension; no private API calls",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "snapshot_ms": frozen_ms,
            "server_snapshot_utc": pd.Timestamp(frozen_ms, unit="ms", tz="UTC").isoformat(),
            "parent_snapshot_sha256": original_hash, "parent_cache": str(original_cache.resolve()),
            "symbols": list(symbols), "frames": {}, "funding": {}, "chunks": {},
            "chronological_periods": chronological_periods(frozen_ms, history_days), "status": "fetching",
            "mode": mode, "requested_history_days": history_days,
            "execution_resolution": "5m" if mode == "m5_only" else "1m",
            "old_period_source_scalp_M1_unavailable": mode == "m5_only",
            "selection_bias_note": "Five current selected contracts; no historical walk-forward universe reconstruction"}
    previous_symbols = set(manifest["symbols"])
    manifest["symbols"] = list(dict.fromkeys([*manifest["symbols"], *symbols]))
    manifest["requested_symbols"] = list(dict.fromkeys([*manifest.get("requested_symbols", manifest["symbols"]), *symbols]))
    manifest["status"] = "fetching"
    write_lock = asyncio.Lock()
    async def save():
        async with write_lock:
            atomic_json(path, manifest)
    own_client = client is None
    if own_client:
        config = Settings(_env_file=None, mexc_api_key="", mexc_api_secret="", auto_execution=False)
        client = MEXCClient(config)
    semaphore = asyncio.Semaphore(concurrency)

    async def extend(symbol, timeframe, recorded):
        original_file = original_cache / f"{symbol}_{timeframe}.csv"
        verify_file(original_file, recorded)
        original = load_frame(original_file)
        validate_frame(original, timeframe, frozen_ms)
        output = cache / original_file.name
        existing = manifest["frames"].setdefault(symbol, {}).get(timeframe)
        if existing:
            verify_file(output, existing)
            validate_frame(load_frame(output), timeframe, frozen_ms)
            return
        if timeframe not in extension_days:
            shutil.copyfile(original_file, output)
            manifest["frames"][symbol][timeframe] = {**recorded, "origin": "verified_original_copy",
                "parent_sha256": recorded["sha256"], "source": PUBLIC_SOURCE,
                "role": "legacy_fine_resolution_control_only" if timeframe == "1m" and mode == "m5_only" else "known_context"}
            await save()
            return
        step_ms = SECONDS[timeframe] * 1000
        desired_ms = (frozen_ms - extension_days[timeframe] * 86_400_000) // step_ms * step_ms - CONTEXT_BARS * step_ms
        final_boundary = frozen_ms // step_ms * step_ms
        cursor = desired_ms
        original_start = int(original.index[0].timestamp() * 1000)
        chunks = []
        while cursor < original_start:
            boundary = min(cursor + CHUNK_DAYS[timeframe] * 86_400_000, original_start)
            count = (boundary - cursor) // step_ms
            key = f"{symbol}_{timeframe}_{cursor}_{boundary}"
            chunk_file = cache / "chunks" / f"{key}.csv"
            info = manifest["chunks"].get(key)
            if info:
                verify_file(chunk_file, info)
                frame = load_frame(chunk_file)
            else:
                print(f"Fetching public {symbol} {timeframe}: {count:,} bars ending "
                    f"{pd.Timestamp(boundary, unit='ms', tz='UTC').isoformat()}", flush=True)
                if not 0 < count < 100_000:
                    raise ResearchDataError("Candle chunk exceeds the safe public client limit")
                frame = await client.get_klines(symbol, timeframe, limit=count, start_time=cursor, end_time=boundary)
                validate_frame(frame, timeframe, frozen_ms, start_ms=cursor, boundary_ms=boundary)
                atomic_csv(chunk_file, frame)
                manifest["chunks"][key] = {**validate_frame(frame, timeframe, frozen_ms),
                    "sha256": sha256_file(chunk_file), "start_ms": cursor, "end_ms_exclusive": boundary,
                    "requested_bars": count, "origin": "new_public_request"}
                await save()
            validate_frame(frame, timeframe, frozen_ms, start_ms=cursor, boundary_ms=boundary)
            chunks.append(frame)
            cursor = boundary
        original = original.loc[original.index >= pd.Timestamp(desired_ms, unit="ms", tz="UTC")]
        merged = pd.concat([*chunks, original]) if chunks else original
        metadata = validate_frame(merged, timeframe, frozen_ms, start_ms=desired_ms, boundary_ms=final_boundary)
        # Three independently queried timestamps detect revisions and endpoint/cutoff errors.
        probes = []
        for index in (0, len(merged) // 2, len(merged) - 1):
            stamp = merged.index[index]
            stamp_ms = int(stamp.timestamp() * 1000)
            probe = await client.get_klines(symbol, timeframe, limit=1, start_time=stamp_ms, end_time=stamp_ms + step_ms)
            validate_frame(probe, timeframe, frozen_ms, start_ms=stamp_ms, boundary_ms=stamp_ms + step_ms)
            columns = ["open", "high", "low", "close", "volume"]
            if not np.allclose(probe.iloc[0][columns].to_numpy(dtype=float), merged.iloc[index][columns].to_numpy(dtype=float), rtol=1e-10, atol=1e-10):
                raise ResearchDataError(f"Public historical data revised at {symbol} {timeframe} {stamp.isoformat()}")
            probes.append({"open_utc": stamp.isoformat(), "close_utc": (stamp + pd.Timedelta(milliseconds=step_ms)).isoformat(),
                "ohlcv_sha256": hashlib.sha256(json.dumps([float(x) for x in probe.iloc[0][columns]], separators=(",", ":")).encode()).hexdigest(),
                "matches_cached_data": True})
        atomic_csv(output, merged)
        manifest["frames"][symbol][timeframe] = {**metadata, "sha256": sha256_file(output),
            "parent_sha256": recorded["sha256"], "requested_bars": len(merged), "context_bars": CONTEXT_BARS,
            "origin": "original_plus_verified_public_chunks", "timestamp_stability_probes": probes}
        await save()
        print(f"Research {symbol} {timeframe} complete: {len(merged):,} gap-free closed bars", flush=True)

    async def fetch_native_symbol(symbol):
        if mode != "m5_only":
            raise ResearchDataError("New native assets require an explicit M5-only diagnostic")
        contract = await client.get_contract(symbol)
        if contract.get("apiAllowed") is not True or contract.get("state") != 0 or contract.get("futureType") != 1 or contract.get("quoteCoin") != "USDT" or contract.get("settleCoin") != "USDT":
            raise ResearchDataError("Native diagnostic contract is not an API-enabled USDT perpetual")
        fields = ("symbol", "baseCoin", "quoteCoin", "settleCoin", "apiAllowed", "state", "futureType", "contractSize", "priceUnit", "openingTime", "createTime", "makerFeeRate", "takerFeeRate")
        manifest.setdefault("contracts", {})[symbol] = {key: contract.get(key) for key in fields}
        limits = {"5m": history_days * 288 + CONTEXT_BARS, "1h": history_days * 24 + CONTEXT_BARS,
            "1d": 600, "1m": 30 * 1440 + CONTEXT_BARS}
        frames = manifest["frames"].setdefault(symbol, {})
        for tf, limit in limits.items():
            output = cache / f"{symbol}_{tf}.csv"
            existing = frames.get(tf)
            if existing:
                verify_file(output, existing)
                frame = load_frame(output)
            else:
                print(f"Fetching native public {symbol} {tf}: up to {limit:,} bars", flush=True)
                frame = await client.get_klines(symbol, tf, limit=limit, end_time=frozen_ms)
            metadata = validate_frame(frame, tf, frozen_ms)
            if tf in {"5m", "1h"}:
                step_ms = SECONDS[tf] * 1000
                boundary = frozen_ms // step_ms * step_ms
                start_ms = boundary - limit * step_ms
                validate_frame(frame, tf, frozen_ms, start_ms=start_ms, boundary_ms=boundary)
            elif tf == "1d":
                required_context_start = pd.Timestamp(frozen_ms, unit="ms", tz="UTC") - pd.Timedelta(days=history_days + CONTEXT_BARS)
                if frame.index[0] > required_context_start:
                    raise ResearchDataError("Native daily context history is insufficient")
            if not existing:
                atomic_csv(output, frame)
                frames[tf] = {**metadata, "sha256": sha256_file(output), "requested_bars": limit,
                    "origin": "new_public_native_asset", "role": "legacy_fine_resolution_control_only" if tf == "1m" else "research_data"}
                await save()
            if tf == "5m" and "timestamp_stability_probes" not in frames[tf]:
                probes = []
                for index in (0, len(frame) // 2, len(frame) - 1):
                    stamp = frame.index[index]
                    stamp_ms = int(stamp.timestamp() * 1000)
                    probe = await client.get_klines(symbol, tf, limit=1, start_time=stamp_ms, end_time=stamp_ms + 300_000)
                    validate_frame(probe, tf, frozen_ms, start_ms=stamp_ms, boundary_ms=stamp_ms + 300_000)
                    columns = ["open", "high", "low", "close", "volume"]
                    if not np.allclose(probe.iloc[0][columns].to_numpy(dtype=float), frame.iloc[index][columns].to_numpy(dtype=float), rtol=1e-10, atol=1e-10):
                        raise ResearchDataError("Native public history changed between timestamp probes")
                    probes.append({"open_utc": stamp.isoformat(), "matches_cached_data": True})
                frames[tf]["timestamp_stability_probes"] = probes
                await save()
        info = manifest["funding"].get(symbol)
        output = cache / f"{symbol}_funding.csv"
        if info:
            verify_file(output, info)
        else:
            since = pd.Timestamp(frozen_ms, unit="ms", tz="UTC") - pd.Timedelta(days=history_days, hours=8)
            rates = await client.get_funding_history(symbol, int(since.timestamp() * 1000))
            rates = rates.loc[rates.index <= pd.Timestamp(frozen_ms, unit="ms", tz="UTC")]
            if rates.empty or rates.index.has_duplicates or not rates.index.is_monotonic_increasing or not np.isfinite(rates.to_numpy(dtype=float)).all():
                raise ResearchDataError("Native funding history is missing or invalid")
            if rates.index[0] > since + pd.Timedelta(hours=8) or rates.index.to_series().diff().dropna().gt(pd.Timedelta(hours=8)).any():
                raise ResearchDataError("Native funding history has insufficient settlement coverage")
            atomic_csv(output, rates, "settlement_utc")
            manifest["funding"][symbol] = {"status": "available", "sha256": sha256_file(output),
                "settlements": len(rates), "first_settlement_utc": rates.index[0].isoformat(),
                "last_settlement_utc": rates.index[-1].isoformat(),
                "maximum_gap_hours": rates.index.to_series().diff().dt.total_seconds().max() / 3600,
                "origin": "new_public_native_asset", "source": "https://www.mexc.com/api-docs/futures/market-endpoints/get-funding-rate-history"}
            await save()

    async def fetch_symbol(symbol):
        async with semaphore:
            if symbol not in parent["frames"]:
                try:
                    await fetch_native_symbol(symbol)
                    manifest.setdefault("excluded_symbols", {}).pop(symbol, None)
                except ResearchDataError as error:
                    manifest.setdefault("excluded_symbols", {})[symbol] = str(error)
                    await save()
                    print(f"Native asset excluded from research: {symbol}: {error}", flush=True)
                return
            for timeframe, recorded in parent["frames"][symbol].items():
                await extend(symbol, timeframe, recorded)
            info = parent.get("funding", {}).get(symbol)
            if not info or info.get("status") != "available":
                raise ResearchDataError(f"Verified original funding unavailable for {symbol}")
            source_file = original_cache / f"{symbol}_funding.csv"
            verify_file(source_file, info)
            rates = load_frame(source_file)
            values = rates["funding_rate"].to_numpy(dtype=float)
            if rates.empty or rates.index.has_duplicates or not rates.index.is_monotonic_increasing or not np.isfinite(values).all():
                raise ResearchDataError(f"Invalid original funding data for {symbol}")
            if rates.index[-1] > pd.Timestamp(frozen_ms, unit="ms", tz="UTC"):
                raise ResearchDataError("Funding settlement occurs after the frozen cutoff")
            earliest_needed = pd.Timestamp(frozen_ms, unit="ms", tz="UTC") - pd.Timedelta(days=history_days)
            if rates.index[0] > earliest_needed or rates.index.to_series().diff().dropna().gt(pd.Timedelta(hours=8)).any():
                raise ResearchDataError("Original funding history has insufficient causal coverage")
            cached = manifest["funding"].get(symbol)
            if cached:
                verify_file(cache / source_file.name, cached)
            else:
                shutil.copyfile(source_file, cache / source_file.name)
                manifest["funding"][symbol] = {**info, "origin": "verified_original_copy", "parent_sha256": info["sha256"]}
            await save()

    tasks = []
    try:
        await save()
        tasks = [asyncio.create_task(fetch_symbol(symbol)) for symbol in symbols]
        await asyncio.gather(*tasks)
        if sha256_file(original_path) != original_hash:
            raise ResearchDataError("Original manifest changed during research fetching")
        manifest["status"] = "complete"
        manifest["symbols"] = [symbol for symbol in manifest["requested_symbols"]
            if symbol in manifest["frames"] and symbol in manifest["funding"] and symbol not in manifest.get("excluded_symbols", {})]
        if "completed_at_utc" not in manifest or set(manifest["symbols"]) != previous_symbols:
            manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        await save()
        return manifest
    except Exception as error:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        manifest["status"] = "incomplete"
        manifest["error"] = str(error)
        await save()
        raise
    finally:
        if own_client:
            await client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-cache", default=".cache/suite")
    parser.add_argument("--cache", default=".cache/research")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--symbol", action="append", choices=(*SYMBOLS, "BTC_USDT", "EUR_USDT", "GBP_USDT", "JPY_USDT"))
    parser.add_argument("--mode", choices=("full", "m5_only"), default="full")
    parser.add_argument("--history-days", type=int, default=365)
    args = parser.parse_args()
    asyncio.run(fetch_research_snapshot(args.original_cache, args.cache,
        symbols=tuple(args.symbol or SYMBOLS), concurrency=args.concurrency,
        mode=args.mode, history_days=args.history_days))


if __name__ == "__main__":
    main()
