"""Public-data CLI boundary checks; no real credentials or network calls."""
import asyncio
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from app import replay as replay_module


NOW = pd.Timestamp("2026-10-05T07:15:00Z")
SYMBOLS = ["ZEC_USDT", "SOL_USDT", "DOGE_USDT", "XRP_USDT", "ETH_USDT"]


def _args(tmp_path, **overrides):
    values = dict(mode="intraday", swing_context="1d", symbol=None, days=30,
                  out=str(tmp_path / "nested" / "report.json"), cache=str(tmp_path / "cache"))
    values.update(overrides)
    return SimpleNamespace(**values)


def _public_boundary(monkeypatch, *, fail_selection=False):
    state = {"klines": [], "funding": [], "replays": [], "selectors": []}

    class Client:
        def __init__(self, settings):
            state["settings"] = settings
            state["client_closed"] = False

        async def get_server_time(self):
            return int(NOW.timestamp() * 1000)

        async def get_klines(self, symbol, tf, limit, end_time):
            state["klines"].append((symbol, tf, limit, end_time))
            return pd.DataFrame({"open": [100.], "high": [101.], "low": [99.], "close": [100.]},
                                index=pd.DatetimeIndex([NOW - pd.Timedelta(seconds=replay_module.SECONDS[tf])], name="timestamp"))

        async def get_funding_history(self, symbol, since_ms):
            state["funding"].append((symbol, since_ms))
            # Public endpoint may include a future settlement: the frozen replay must omit it.
            return pd.Series([.001, .002], index=pd.DatetimeIndex([NOW, NOW + pd.Timedelta(hours=8)]), name="funding_rate")

        async def close(self):
            state["client_closed"] = True

    class Selector:
        def __init__(self, client, settings):
            state["selectors"].append(self)
            self.closed = False
            self.snapshot = {"version": 1, "selected": [{"symbol": s} for s in SYMBOLS], "selected_at": NOW.isoformat()}
            assert settings.pair_count == 5
            assert settings.pair_selection == "dynamic"

        async def select(self):
            if fail_selection:
                raise RuntimeError("Public market-cap source unavailable")
            return self.snapshot["selected"]

        async def close(self):
            self.closed = True

    def run_replay(symbol, frames, settings, start_at, funding_rates):
        state["replays"].append((symbol, frames, start_at, funding_rates))
        return {"symbol": symbol, "trades": [], "funding_total_usdt": -.1,
                "final_realized_equity": 999.9, "unrealized_pnl_usdt": 2., "marked_equity": 1001.9}

    monkeypatch.setattr(replay_module, "MEXCClient", Client)
    monkeypatch.setattr(replay_module, "UniverseSelector", Selector)
    monkeypatch.setattr(replay_module, "replay", run_replay)
    monkeypatch.setenv("PAIR_SELECTION", "dynamic")
    return state


def test_default_replay_selects_five_and_passes_only_public_frozen_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MEXC_API_KEY=synthetic-file-access\nMEXC_API_SECRET=synthetic-file-secret\nAUTO_EXECUTION=true\n")
    monkeypatch.setenv("MEXC_API_KEY", "synthetic-environment-access")
    monkeypatch.setenv("MEXC_API_SECRET", "synthetic-environment-secret")
    monkeypatch.setenv("AUTO_EXECUTION", "true")
    state = _public_boundary(monkeypatch)
    args = _args(tmp_path)
    asyncio.run(replay_module.run(args))

    assert state["settings"].mexc_api_key == state["settings"].mexc_api_secret == ""
    assert state["settings"].auto_execution is False
    assert [r[0] for r in state["replays"]] == SYMBOLS
    assert len(state["klines"]) == 5 * 3
    assert all(call[3] == int(NOW.timestamp() * 1000) for call in state["klines"])
    since = NOW - pd.Timedelta(days=30)
    assert state["funding"] == [(s, int(since.timestamp() * 1000)) for s in SYMBOLS]
    for _, frames, start_at, funding in state["replays"]:
        assert set(frames) == {"1d", "1h", "5m"}
        assert start_at == since
        assert funding.index.tolist() == [NOW]
    assert state["client_closed"] and state["selectors"][0].closed
    report = json.loads((tmp_path / "nested/report.json").read_text())
    assert report["symbols"] == SYMBOLS
    assert report["universe_selection"] == state["selectors"][0].snapshot
    assert report["fetched_at"] == NOW.isoformat()
    assert report["assumptions"]["actual_funding_rates"] is True
    assert report["results"][0]["marked_equity"] == 1001.9
    assert "synthetic" not in json.dumps(report)
    assert len(list((tmp_path / "cache").glob("*.csv"))) == 20


def test_explicit_symbols_skip_selector_and_weekly_frames_follow_mode(monkeypatch, tmp_path):
    state = _public_boundary(monkeypatch)
    args = _args(tmp_path, symbol=["BTC_USDT", "BTC_USDT", "ETH_USDT"], mode="swing", swing_context="1w")
    asyncio.run(replay_module.run(args))
    assert state["selectors"] == []
    assert [row[0] for row in state["replays"]] == ["BTC_USDT", "ETH_USDT"]
    assert all(set(row[1]) == {"1w", "4h"} for row in state["replays"])
    report = json.loads((tmp_path / "nested/report.json").read_text())
    assert report["universe_selection"] is None
    assert state["client_closed"]


def test_selection_failure_closes_both_public_clients_without_replaying(monkeypatch, tmp_path):
    state = _public_boundary(monkeypatch, fail_selection=True)
    with pytest.raises(RuntimeError, match="market-cap source"):
        asyncio.run(replay_module.run(_args(tmp_path)))
    assert state["client_closed"] and state["selectors"][0].closed
    assert state["klines"] == state["funding"] == state["replays"] == []
    assert not (tmp_path / "nested/report.json").exists()
