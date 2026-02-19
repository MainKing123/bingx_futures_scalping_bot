import asyncio

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
