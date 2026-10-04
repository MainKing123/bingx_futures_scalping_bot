import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from app.external_data import (ExternalDataError, aggregate_minutes, archive_csv,
                               months, parse_funding, parse_klines, verify_checksum, validate_complete_manifest,
                               atomic_csv)


def minute_csv(start="2024-01-01", rows=5):
    first = pd.Timestamp(start, tz="UTC")
    values = []
    for i in range(rows):
        opened = int((first + pd.Timedelta(minutes=i)).timestamp() * 1000)
        values.append([opened, 100 + i, 102 + i, 99 + i, 101 + i, 2,
                       opened + 59999, 200, 10, 1, 100, 0])
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    return pd.DataFrame(values, columns=columns)


def test_official_checksum_rejects_modified_archive_and_wrong_name():
    data = b"archive bytes"
    checksum = hashlib.sha256(data).hexdigest() + "  BTCUSDT-1m-2024-01.zip\n"
    assert verify_checksum(data, checksum, "BTCUSDT-1m-2024-01.zip") == hashlib.sha256(data).hexdigest()
    with pytest.raises(ExternalDataError, match="mismatch"):
        verify_checksum(data + b"tamper", checksum, "BTCUSDT-1m-2024-01.zip")
    with pytest.raises(ExternalDataError, match="record"):
        verify_checksum(data, checksum, "ETHUSDT-1m-2024-01.zip")


def test_archive_member_is_read_in_memory_and_rejects_unexpected_paths():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("sample.csv", "csv bytes")
    assert archive_csv(buffer.getvalue(), "sample.zip") == b"csv bytes"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../sample.csv", "csv bytes")
    with pytest.raises(ExternalDataError, match="member"):
        archive_csv(buffer.getvalue(), "sample.zip")


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unordered", "microseconds", "bad_close", "invalid_geometry", "nan"])
def test_observed_minute_parser_rejects_data_faults(fault):
    data = minute_csv()
    if fault == "missing":
        data = data.drop(index=2)
    elif fault == "duplicate":
        data.loc[2, "open_time"] = data.loc[1, "open_time"]
        data.loc[2, "close_time"] = data.loc[1, "close_time"]
    elif fault == "unordered":
        data = data.iloc[[0, 2, 1, 3, 4]]
    elif fault == "microseconds":
        data["open_time"] *= 1000
        data["close_time"] *= 1000
    elif fault == "bad_close":
        data.loc[2, "close_time"] += 1
    elif fault == "invalid_geometry":
        data.loc[2, "high"] = 10
    else:
        data.loc[2, "close"] = np.nan
    with pytest.raises((ExternalDataError, ValueError)):
        parse_klines(data.to_csv(index=False).encode(), "1m",
                     pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-01T00:05Z"))


def test_aggregation_preserves_actual_ohlcv_and_drops_incomplete_groups():
    frame = parse_klines(minute_csv(rows=10).to_csv(index=False).encode(), "1m",
                         pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-01T00:10Z"))
    result = aggregate_minutes(frame, "5m")
    assert result.iloc[0].to_dict() == {"open": 100, "high": 106, "low": 99, "close": 105, "volume": 10}
    incomplete = aggregate_minutes(frame.drop(frame.index[2]), "5m")
    assert len(incomplete) == 1
    assert incomplete.index[0] == pd.Timestamp("2024-01-01T00:05Z")


def test_exact_funding_jitter_is_preserved_and_missing_event_is_rejected():
    start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")
    times = [int(start.timestamp() * 1000), int((start + pd.Timedelta(hours=8)).timestamp() * 1000) + 13,
             int((start + pd.Timedelta(hours=16)).timestamp() * 1000)]
    data = pd.DataFrame({"calc_time": times, "funding_interval_hours": [8, 8, 8],
                         "last_funding_rate": [.0001, -.0002, .0003]})
    rates = parse_funding(data.to_csv(index=False).encode(), start, end)
    assert rates.index[1].microsecond == 13000
    assert rates.iloc[1] == -.0002
    with pytest.raises(ExternalDataError, match="settlements"):
        parse_funding(data.iloc[:2].to_csv(index=False).encode(), start, end)
    data.loc[1, "funding_interval_hours"] = 4
    with pytest.raises(ExternalDataError, match="cadence"):
        parse_funding(data.to_csv(index=False).encode(), start, end)


def test_calendar_months_include_leap_february_without_partial_end_month():
    assert months(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-04-01", tz="UTC")) == ["2024-01", "2024-02", "2024-03"]


def test_duplicate_minutes_cannot_disguise_missing_minute_in_complete_bin():
    frame = parse_klines(minute_csv().to_csv(index=False).encode(), "1m",
                         pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-01T00:05Z"))
    defective = frame.iloc[[0, 1, 1, 3, 4]]
    with pytest.raises(ExternalDataError, match="Duplicate"):
        aggregate_minutes(defective, "5m")
    off_grid = frame.copy()
    off_grid.index = off_grid.index + pd.Timedelta(seconds=1)
    with pytest.raises(ExternalDataError, match="off-grid"):
        aggregate_minutes(off_grid, "5m")


def test_complete_flag_without_registered_sample_is_rejected():
    with pytest.raises(ExternalDataError, match="structure"):
        validate_complete_manifest({"status": "complete", "frames": {}, "funding": {}})


def test_fractional_funding_csv_roundtrips_default_replay_date_parser(tmp_path):
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2024-01-01T08:00:00.013Z")])
    source = pd.DataFrame({"funding_rate": [.0001, -.0002]}, index=index)
    path = tmp_path / "funding.csv"
    atomic_csv(path,source,date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    loaded = pd.read_csv(path,index_col=0)
    loaded.index = pd.to_datetime(loaded.index,utc=True)
    assert loaded.index.equals(source.index)
    assert loaded.funding_rate.tolist() == source.funding_rate.tolist()
