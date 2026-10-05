import asyncio
import pytest
from app.runtime_settings import RuntimeSettings
from app.universe import UniverseSelectionError
from test_universe import create_selector


def test_core_btc_eth_plus_three_highest_eligible_volatility_pairs(tmp_path,monkeypatch):
    async def run():
        client,selector=create_selector(tmp_path,monkeypatch,settings=RuntimeSettings(_env_file=None))
        try:
            rows=await selector.select()
            assert [row["symbol"] for row in rows]==["BTC_USDT","ETH_USDT","FFF_USDT","EEE_USDT","DDD_USDT"]
            assert len(rows)==5
            assert selector.snapshot["parameters"]["core_symbols"]==["BTC_USDT","ETH_USDT"]
        finally:
            await selector.close()
    asyncio.run(run())


def test_ineligible_core_pair_is_not_silently_replaced(tmp_path,monkeypatch):
    async def run():
        client,selector=create_selector(tmp_path,monkeypatch,settings=RuntimeSettings(_env_file=None))
        client.tickers[0]["quoteVolume"]=1
        try:
            with pytest.raises(UniverseSelectionError,match="Required core"):
                await selector.select()
        finally:
            await selector.close()
    asyncio.run(run())
