import asyncio
from datetime import datetime, timezone

import pandas as pd
from app.config import Settings
from app.exchange.client import BingXClient


def test_usdt_filtering_in_top_symbols_and_contracts():
    async def run():
        client = BingXClient(Settings())

        async def fake_request(method, path, params=None, signed=False):
            return [
                {"symbol": "BTC-USDT", "quoteVolume": "20000000"},
                {"symbol": "ETH-USDT", "quoteVolume": "15000000"},
                {"symbol": "BUSD-USDT", "quoteVolume": "99999999"},
                {"symbol": "BTC-USDC", "quoteVolume": "30000000"},
            ]

        client._request = fake_request
        symbols = await client.get_top_symbols(limit=10, min_volume_usd=1)
        contracts = await client.get_contracts()

        assert "BTC-USDT" in symbols
        assert "BUSD-USDT" not in symbols
        assert all(item["symbol"].endswith("-USDT") for item in contracts)
        assert all(item["symbol"] != "BUSD-USDT" for item in contracts)
        await client.close()

    asyncio.run(run())


def test_top_volatile_symbols_ranking():
    async def run():
        client = BingXClient(Settings())

        async def fake_request(method, path, params=None, signed=False):
            return [
                {"symbol": "BTC-USDT", "quoteVolume": "20000000"},
                {"symbol": "ETH-USDT", "quoteVolume": "18000000"},
                {"symbol": "XRP-USDT", "quoteVolume": "17000000"},
                {"symbol": "BUSD-USDT", "quoteVolume": "99999999"},
            ]

        def build_frame(spread_pct: float) -> pd.DataFrame:
            idx = pd.date_range(datetime(2025, 1, 1, tzinfo=timezone.utc), periods=10, freq="min")
            close = pd.Series([100.0] * 10, index=idx)
            high = close * (1 + spread_pct / 200)
            low = close * (1 - spread_pct / 200)
            return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 1000}, index=idx)

        async def fake_klines(symbol, interval, limit=500, start_time=None, end_time=None):
            spreads = {
                "BTC-USDT": 1.8,
                "ETH-USDT": 0.8,
                "XRP-USDT": 2.5,
            }
            return build_frame(spreads[symbol])

        client._request = fake_request
        client.get_klines = fake_klines

        ranked = await client.get_top_volatile_symbols(limit=2, min_volume_usd=1, pool_size=5, interval="1m", lookback=10)

        assert len(ranked) == 2
        assert ranked[0]["symbol"] == "XRP-USDT"
        assert ranked[1]["symbol"] == "BTC-USDT"
        assert ranked[0]["volatility"] > ranked[1]["volatility"]
        await client.close()

    asyncio.run(run())
