"""Separate Binance USD-M research data; never a substitute for MEXC fills.

Public monthly archives are verified against their official SHA256 files.
Minutes are observed candles, while higher frames are explicitly aggregated.
No replay or private exchange request is performed by this module.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/monthly"
SOURCE = "https://github.com/binance/binance-public-data"
SYMBOLS = {"BTC_USDT": "BTCUSDT", "ETH_USDT": "ETHUSDT"}
START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2026-01-01", tz="UTC")
WARMUP_START = pd.Timestamp("2022-01-01", tz="UTC")
SECONDS = {"1m": 60, "5m": 300, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}


class ExternalDataError(RuntimeError):
    pass


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path, frame, *, date_format=None):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index_label="timestamp", date_format=date_format)
    temporary.replace(path)


def months(start, end):
    return [value.strftime("%Y-%m") for value in pd.date_range(start, end, freq="MS", inclusive="left")]


def verify_checksum(data, checksum, filename):
    fields = checksum.strip().split()
    if len(fields) != 2 or fields[1].lstrip("*") != filename or len(fields[0]) != 64:
        raise ExternalDataError("Invalid official archive checksum record")
    if sha256(data) != fields[0].lower():
        raise ExternalDataError(f"Archive checksum mismatch: {filename}")
    return fields[0].lower()


def archive_csv(data, filename):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        expected = filename.removesuffix(".zip") + ".csv"
        if archive.namelist() != [expected]:
            raise ExternalDataError("Unexpected archive member; extraction refused")
        return archive.read(expected)


def parse_klines(csv_data, timeframe, start, end):
    """Validate exact expected OPEN-time observations, including close time."""
    header = csv_data.splitlines()[0].decode("utf-8-sig").split(",")[0]
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    raw = pd.read_csv(io.BytesIO(csv_data), header=0 if header == "open_time" else None,
                      names=columns)
    if raw.empty:
        raise ExternalDataError("Empty archive candles")
    raw = raw.apply(pd.to_numeric, errors="raise")
    step_ms = SECONDS[timeframe] * 1000
    if not (raw.close_time == raw.open_time + step_ms - 1).all():
        raise ExternalDataError("Unexpected USD-M timestamp units or candle close time")
    index = pd.to_datetime(raw.open_time, unit="ms", utc=True)
    expected = pd.date_range(start, end, freq=pd.Timedelta(seconds=SECONDS[timeframe]), inclusive="left")
    if not pd.DatetimeIndex(index).equals(expected):
        raise ExternalDataError("Archive missing, duplicate, unordered or out-of-window candles")
    frame = raw[["open", "high", "low", "close", "volume"]].copy()
    frame.index = pd.DatetimeIndex(index, name="timestamp")
    validate_values(frame)
    return frame


def validate_values(frame):
    fields = ["open", "high", "low", "close", "volume"]
    data = frame[fields].to_numpy(dtype=float)
    if not np.isfinite(data).all() or (data[:, :4] <= 0).any() or (data[:, 4] < 0).any():
        raise ExternalDataError("Nonfinite or nonpositive archive OHLCV")
    if frame.high.lt(frame[["open", "close", "low"]].max(axis=1)).any() or frame.low.gt(frame[["open", "close", "high"]].min(axis=1)).any():
        raise ExternalDataError("Invalid archive OHLC geometry")


def parse_funding(csv_data, start, end):
    raw = pd.read_csv(io.BytesIO(csv_data))
    required = {"calc_time", "funding_interval_hours", "last_funding_rate"}
    if not required.issubset(raw) or raw.empty:
        raise ExternalDataError("Missing public Binance funding columns")
    raw = raw.apply(pd.to_numeric, errors="raise")
    index = pd.DatetimeIndex(pd.to_datetime(raw.calc_time, unit="ms", utc=True))
    if index.has_duplicates or not index.is_monotonic_increasing or not np.isfinite(raw.last_funding_rate).all():
        raise ExternalDataError("Invalid public funding observations")
    if not raw.funding_interval_hours.eq(8).all():
        raise ExternalDataError("Funding cadence differs from this preregistered BTC/ETH sample")
    expected = pd.date_range(start, end, freq="8h", inclusive="left")
    # The exchange records occasional millisecond settlement jitter. Keep the
    # exact event timestamp; tolerate only a <60s delay when auditing coverage.
    if len(index) != len(expected) or not index.floor("min").equals(expected):
        raise ExternalDataError("Funding history has missing or unexpected settlements")
    return pd.Series(raw.last_funding_rate.to_numpy(dtype=float), index=index, name="funding_rate")


def aggregate_minutes(frame, timeframe):
    """Aggregate only COMPLETE higher bars; missing minutes are never filled."""
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None or frame.empty:
        raise ExternalDataError("Aggregation requires nonempty timezone-aware observed minutes")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing or not frame.index.equals(frame.index.floor("min")):
        raise ExternalDataError("Duplicate, unordered or off-grid observed minutes")
    validate_values(frame)
    if timeframe == "1m":
        return frame.copy()
    frequency = {"5m": "5min", "1h": "1h", "4h": "4h", "1d": "1D", "1w": "W-MON"}[timeframe]
    groups = frame.resample(frequency, label="left", closed="left", origin="epoch")
    result = groups.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    expected = SECONDS[timeframe] // 60
    counts = groups["close"].count()
    result = result.loc[counts.eq(expected)]
    validate_values(result)
    return result


def metadata(frame, timeframe):
    step = pd.Timedelta(seconds=SECONDS[timeframe])
    gaps = frame.index.to_series().diff().dropna()
    missing = int(sum(max(0, int(value / step) - 1) for value in gaps))
    return {"bars": len(frame), "first_open_utc": frame.index[0].isoformat(),
            "last_open_utc": frame.index[-1].isoformat(),
            "last_close_utc": (frame.index[-1] + step).isoformat(), "missing_intervals": missing}


def validate_complete_manifest(manifest):
    """A complete flag cannot stand in for the registered sample structure."""
    mandatory = {"1m", "5m", "1h", "4h", "1d"}
    if (manifest.get("venue") != "binance_usdm" or manifest.get("execution_resolution") != "1m"
            or manifest.get("symbols") != list(SYMBOLS)
            or set(manifest.get("frames", {})) != set(SYMBOLS)
            or set(manifest.get("funding", {})) != set(SYMBOLS)
            or manifest.get("server_snapshot_utc") != END.isoformat()
            or manifest.get("outcomes_start_utc") != START.isoformat()
            or manifest.get("warmup_start_utc") != WARMUP_START.isoformat()
            or manifest.get("snapshot_ms") != int(END.timestamp() * 1000)):
        raise ExternalDataError("Incomplete or different registered external sample structure")
    for symbol in SYMBOLS:
        if set(manifest["frames"][symbol]) != mandatory:
            raise ExternalDataError("Missing mandatory external timeframe")
        for timeframe, record in manifest["frames"][symbol].items():
            start = WARMUP_START if timeframe == "1d" else START
            expected = int((END - start).total_seconds() // SECONDS[timeframe])
            if (record.get("bars") != expected or record.get("missing_intervals") != 0
                    or record.get("first_open_utc") != start.isoformat()
                    or record.get("last_close_utc") != END.isoformat()
                    or record.get("filename") != f"{symbol}_{timeframe}.csv"
                    or len(record.get("sha256", "")) != 64):
                raise ExternalDataError("Incorrect external candle coverage or filename")
        funding = manifest["funding"][symbol]
        if (funding.get("records") != int((END - START).total_seconds() // (8 * 3600))
                or funding.get("filename") != f"{symbol}_funding.csv"
                or funding.get("cadence_hours") != 8 or len(funding.get("sha256", "")) != 64):
            raise ExternalDataError("Incomplete external funding record")


async def download_archive(client, semaphore, cache, kind, symbol, month, timeframe=None):
    filename = f"{symbol}-{timeframe}-{month}.zip" if kind == "klines" else f"{symbol}-fundingRate-{month}.zip"
    url = f"{BASE}/klines/{symbol}/{timeframe}/{filename}" if kind == "klines" else f"{BASE}/fundingRate/{symbol}/{filename}"
    path = cache / "archives" / filename
    checksum_path = path.with_suffix(".zip.CHECKSUM")
    async with semaphore:
        if path.exists() and checksum_path.exists():
            data, checksum = path.read_bytes(), checksum_path.read_text(encoding="utf-8")
        else:
            response = await client.get(url)
            response.raise_for_status()
            checksum_response = await client.get(url + ".CHECKSUM")
            checksum_response.raise_for_status()
            data, checksum = response.content, checksum_response.text
            verify_checksum(data, checksum, filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".zip.tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
            checksum_path.write_text(checksum, encoding="utf-8")
        digest = verify_checksum(data, checksum, filename)
    return kind, timeframe, month, archive_csv(data, filename), {
        "url": url, "checksum_url": url + ".CHECKSUM", "filename": filename,
        "sha256": digest, "bytes": len(data)}


async def fetch_external_snapshot(cache=".cache/binance_core", concurrency=4):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "snapshot.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            validate_complete_manifest(existing)
            for record in [value for frames in existing["frames"].values() for value in frames.values()] + list(existing["funding"].values()):
                path = cache / record["filename"]
                if not path.exists() or sha256(path.read_bytes()) != record["sha256"]:
                    raise ExternalDataError("Modified external research cache; frozen snapshot refused")
            return existing
    manifest = {"source": "Binance public USD-M futures archives", "source_url": SOURCE,
                "venue": "binance_usdm", "status": "incomplete", "symbols": list(SYMBOLS),
                "server_snapshot_utc": END.isoformat(), "snapshot_ms": int(END.timestamp() * 1000),
                "outcomes_start_utc": START.isoformat(), "warmup_start_utc": WARMUP_START.isoformat(),
                "execution_resolution": "1m", "never_described_as_mexc_execution": True,
                "downloaded_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "frames": {}, "funding": {}, "archives": {},
                "volume_units": "base asset", "funding_mark_price_proxy": "last known same-venue execution close"}
    atomic_json(manifest_path, manifest)
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True, limits=httpx.Limits(max_connections=concurrency)) as client:
        for local_symbol, archive_symbol in SYMBOLS.items():
            tasks = [download_archive(client, semaphore, cache, "klines", archive_symbol, month, "1m") for month in months(START, END)]
            tasks += [download_archive(client, semaphore, cache, "klines", archive_symbol, month, "1d") for month in months(WARMUP_START, START)]
            tasks += [download_archive(client, semaphore, cache, "fundingRate", archive_symbol, month) for month in months(START, END)]
            items = await asyncio.gather(*tasks)
            minute_parts, daily_parts, funding_parts = [], [], []
            for kind, timeframe, month, csv_data, record in items:
                left = pd.Timestamp(month + "-01", tz="UTC")
                right = left + pd.offsets.MonthBegin(1)
                manifest["archives"][record["filename"]] = record
                if kind == "fundingRate":
                    funding_parts.append(parse_funding(csv_data, left, right))
                elif timeframe == "1m":
                    minute_parts.append(parse_klines(csv_data, timeframe, left, right))
                else:
                    daily_parts.append(parse_klines(csv_data, timeframe, left, right))
            minutes = pd.concat(minute_parts).sort_index()
            funding = pd.concat(funding_parts).sort_index()
            for timeframe in ("1m", "5m", "1h", "4h", "1d"):
                frame = aggregate_minutes(minutes, timeframe)
                if timeframe == "1d":
                    frame = pd.concat([*daily_parts, frame]).sort_index()
                if frame.index.has_duplicates or not frame.index.to_series().diff().dropna().eq(pd.Timedelta(seconds=SECONDS[timeframe])).all():
                    raise ExternalDataError("Cross-month candle coverage incomplete")
                path = cache / f"{local_symbol}_{timeframe}.csv"
                atomic_csv(path, frame)
                manifest["frames"].setdefault(local_symbol, {})[timeframe] = {
                    **metadata(frame, timeframe), "filename": path.name, "sha256": sha256(path.read_bytes()),
                    "aggregation": "observed M1" if timeframe == "1m" else "complete M1 OHLCV groups; D1 warmup from observed D1 archives"}
            path = cache / f"{local_symbol}_funding.csv"
            # Uniform fractional-second storage prevents pandas 2's inferred
            # format rejecting a later +millisecond settlement in the same CSV.
            atomic_csv(path, funding.to_frame(), date_format="%Y-%m-%dT%H:%M:%S.%f%z")
            manifest["funding"][local_symbol] = {"filename": path.name, "sha256": sha256(path.read_bytes()),
                "records": len(funding), "first_settlement_utc": funding.index[0].isoformat(),
                "last_settlement_utc": funding.index[-1].isoformat(), "cadence_hours": 8,
                "exact_settlement_milliseconds_preserved": True, "actual_funding_rates": True}
            atomic_json(manifest_path, manifest)
            print(f"Verified {local_symbol}: {len(minutes):,} actual minutes, {len(funding):,} actual funding records", flush=True)
    manifest["status"] = "complete"
    validate_complete_manifest(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=".cache/binance_core")
    parser.add_argument("--concurrency", type=int, choices=range(1, 7), default=4)
    args = parser.parse_args()
    asyncio.run(fetch_external_snapshot(args.cache, args.concurrency))


if __name__ == "__main__":
    main()
