import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
import pytest

from app.config import Settings
from app.universe import UniverseSelectionError, UniverseSelector

NOW = datetime(2026, 10, 4, 15, 23, tzinfo=timezone.utc)
SERVER_MS = int(NOW.timestamp() * 1000)


def caps():
    symbols = ["BTC", "ETH", "USDT", "WBTC", "AAA", "BBB", "CCC", "DDD", "EEE", "FFF"] + [f"COIN{i}" for i in range(11, 21)]
    records = [{"id": symbol.lower(), "symbol": symbol.lower(), "name": symbol,
                "market_cap_rank": i + 1, "market_cap": 30_000_000_000 - i * 1_000_000,
                "last_updated": NOW.isoformat()} for i, symbol in enumerate(symbols)]
    records[2]["id"] = "tether"
    records[3]["id"] = "wrapped-bitcoin"
    return records


class FakeMEXC:
    def __init__(self):
        self.calls = []
        self.scores = {"BTC": .1, "ETH": .2, "USDT": 20, "WBTC": 20, "AAA": .3, "BBB": 9,
                       "CCC": 8, "DDD": .6, "EEE": .7, "FFF": .8}
        self.contracts = [{"symbol": f"{s}_USDT", "baseCoin": s, "quoteCoin": "USDT", "settleCoin": "USDT",
                           "apiAllowed": s != "BBB", "state": 0, "futureType": 1} for s in self.scores]
        self.tickers = [{"symbol": f"{s}_USDT", "quoteVolume": 49_000_000 if s == "CCC" else 100_000_000} for s in self.scores]
        self.bad_symbol = None
    async def get_contracts(self):
        self.calls.append("contracts")
        return self.contracts
    async def get_tickers(self):
        self.calls.append("tickers")
        return self.tickers
    async def get_server_time(self):
        return SERVER_MS
    async def get_klines(self, symbol, interval, limit, end_time=None):
        self.calls.append(symbol)
        assert interval == "15m" and limit == 97
        assert end_time == SERVER_MS - 1500
        if symbol == self.bad_symbol:
            raise RuntimeError("No candles")
        score = self.scores[symbol.removesuffix("_USDT")]
        index = pd.date_range(end="2026-10-04T15:00:00Z", periods=97, freq="15min")
        return pd.DataFrame({"high": 100 + score / 2, "low": 100 - score / 2, "close": 100}, index=index)


def create_selector(tmp_path, monkeypatch, *, handler=None, settings=None):
    monkeypatch.setattr("app.universe._utc_now", lambda: NOW)
    client = FakeMEXC()
    def normal(request):
        assert request.url.host == "api.coingecko.com"
        assert request.url.params["order"] == "market_cap_desc"
        assert request.url.params["per_page"] == "100"
        assert "ApiKey" not in request.headers
        return httpx.Response(200, json=caps())
    selector = UniverseSelector(client, settings or Settings(universe_refresh_seconds=3600),
                                cap_transport=httpx.MockTransport(handler or normal), cache_path=tmp_path / "universe.json")
    return client, selector


def test_exact_five_are_top_volatility_large_caps_after_exclusions(tmp_path, monkeypatch):
    async def run():
        client, selector = create_selector(tmp_path, monkeypatch)
        rows = await selector.select()
        assert [r["symbol"] for r in rows] == ["FFF_USDT", "EEE_USDT", "DDD_USDT", "AAA_USDT", "ETH_USDT"]
        assert len(rows) == 5
        assert all(r["market_cap_rank"] <= 20 and r["quoteVolume"] >= 50_000_000 for r in rows)
        assert rows[0]["natr_percent"] == pytest.approx(.8)
        assert "USDT_USDT" not in client.calls and "WBTC_USDT" not in client.calls
        assert "BBB_USDT" not in client.calls and "CCC_USDT" not in client.calls
        assert len(selector.snapshot["candidates"]) == 6
        assert (tmp_path / "universe.json").exists()
        await selector.close()
    asyncio.run(run())


def test_current_snapshot_reuses_selection_without_http_and_cannot_be_mutated_by_caller(tmp_path, monkeypatch):
    async def run():
        client, selector = create_selector(tmp_path, monkeypatch)
        rows = await selector.select()
        count = len(client.calls)
        rows[0]["symbol"] = "BAD_USDT"
        again = await selector.select()
        assert again[0]["symbol"] == "FFF_USDT" and len(client.calls) == count
        await selector.close()
        def unexpected(request):
            raise AssertionError("Fresh persisted selection should not call CoinGecko")
        other = UniverseSelector(client, Settings(universe_refresh_seconds=3600), cap_transport=httpx.MockTransport(unexpected),
                                 cache_path=tmp_path / "universe.json")
        assert [r["symbol"] for r in await other.select()] == [r["symbol"] for r in again]
        assert len(client.calls) == count
        await other.close()
    asyncio.run(run())


def test_429_uses_only_fresh_verified_cap_snapshot_and_reranks_live_data(tmp_path, monkeypatch):
    async def run():
        client, selector = create_selector(tmp_path, monkeypatch)
        await selector.select()
        cap_time = selector.snapshot["market_cap_fetched_at"]
        await selector.cap_client.aclose()
        calls = []
        def failure(request):
            calls.append(request)
            return httpx.Response(429, text="limited")
        selector.cap_client = httpx.AsyncClient(transport=httpx.MockTransport(failure))
        client.scores["BTC"] = 3
        result = await selector.select(force_refresh=True)
        assert result[0]["symbol"] == "BTC_USDT"
        assert selector.snapshot["market_cap_source"] == "cache"
        assert selector.snapshot["market_cap_fetched_at"] == cap_time
        assert len(calls) == 1
        await selector.close()
    asyncio.run(run())


@pytest.mark.parametrize("status", [401, 429, 500])
def test_no_cap_data_fails_closed_without_static_trading_symbol_fallback(tmp_path, monkeypatch, status):
    async def run():
        client, selector = create_selector(tmp_path, monkeypatch, handler=lambda r: httpx.Response(status))
        with pytest.raises(UniverseSelectionError, match="paused"):
            await selector.select()
        assert not client.calls
        await selector.close()
    asyncio.run(run())


def test_expired_selection_and_stale_cap_cache_cannot_be_used(tmp_path, monkeypatch):
    async def run():
        client, selector = create_selector(tmp_path, monkeypatch)
        await selector.select()
        old = (NOW - timedelta(hours=7)).isoformat()
        selector.snapshot["selected_at"] = old
        selector.snapshot["market_cap_fetched_at"] = old
        await selector.cap_client.aclose()
        selector.cap_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
        with pytest.raises(UniverseSelectionError):
            await selector.select()
        await selector.close()
    asyncio.run(run())


def test_missing_rank_or_stale_provider_records_fail_closed(tmp_path, monkeypatch):
    async def run():
        stale = caps()
        for row in stale:
            row["last_updated"] = (NOW - timedelta(hours=7)).isoformat()
        for response in (caps()[:-1], stale):
            client, selector = create_selector(tmp_path, monkeypatch, handler=lambda r: httpx.Response(200, json=response))
            with pytest.raises(UniverseSelectionError):
                await selector.select()
            await selector.close()
    asyncio.run(run())


def test_fewer_than_five_and_any_missing_candidate_candles_pause_selection(tmp_path, monkeypatch):
    async def run():
        client, selector = create_selector(tmp_path, monkeypatch)
        client.contracts = client.contracts[:2]
        with pytest.raises(UniverseSelectionError, match="Fewer than five"):
            await selector.select()
        await selector.close()
        client, selector = create_selector(tmp_path, monkeypatch)
        client.bad_symbol = "BTC_USDT"  # Even a low-ranked missing candidate blocks claiming top five.
        with pytest.raises(UniverseSelectionError, match="Complete fresh volatility"):
            await selector.select()
        await selector.close()
    asyncio.run(run())


def test_true_range_includes_previous_close_and_rejects_gaps_or_live_candles():
    index = pd.date_range(end="2026-10-04T15:00:00Z", periods=97, freq="15min")
    frame = pd.DataFrame({"high": 102.0, "low": 99.0, "close": 100.0}, index=index)
    frame.iloc[0] = [92, 89, 90]  # The next candle's TR is 102-90=12, not its range 3.
    expected = ((12 / 100) + 95 * (3 / 100)) / 96 * 100
    assert UniverseSelector.natr_percent(frame, server_ms=SERVER_MS) == pytest.approx(expected)
    with pytest.raises(UniverseSelectionError):
        UniverseSelector.natr_percent(frame.drop(index[2]), server_ms=SERVER_MS)
    with pytest.raises(UniverseSelectionError, match="stale or unclosed"):
        UniverseSelector.natr_percent(frame, server_ms=SERVER_MS - 900_000)


def test_fixed_mode_requires_exact_five_and_is_explicitly_marked(tmp_path, monkeypatch):
    async def run():
        def unexpected(request):
            raise AssertionError("Fixed mode must not fetch capitalization")
        settings = Settings(pair_selection="fixed", trading_symbols=["BTC_USDT", "ETH_USDT", "AAA_USDT", "DDD_USDT", "EEE_USDT"])
        client, selector = create_selector(tmp_path, monkeypatch, handler=unexpected, settings=settings)
        rows = await selector.select()
        assert [r["symbol"] for r in rows] == settings.trading_symbols
        assert all(r["market_cap_rank"] is None for r in rows)
        assert selector.snapshot["source"] == "explicit_fixed_configuration"
        await selector.close()
        settings = Settings(pair_selection="fixed", trading_symbols=["BTC_USDT", "ETH_USDT"])
        client, selector = create_selector(tmp_path, monkeypatch, handler=unexpected, settings=settings)
        with pytest.raises(UniverseSelectionError, match="exactly five"):
            await selector.select()
        await selector.close()
    asyncio.run(run())
