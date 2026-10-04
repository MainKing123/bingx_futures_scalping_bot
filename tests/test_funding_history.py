import asyncio

import httpx
import pandas as pd
import pytest

from app.config import Settings
from app.exchange.client import MEXCAPIError, MEXCClient


def _client(handler):
    client = MEXCClient(Settings(_env_file=None, mexc_api_key="", mexc_api_secret=""),
                        transport=httpx.MockTransport(handler))

    async def no_throttle(path):
        pass

    client._throttle = no_throttle
    return client


def test_public_funding_paginates_deduplicates_and_stops_at_requested_boundary():
    async def run():
        newest = pd.Timestamp("2026-10-05T00:00:00Z")
        events = [{"settleTime": int((newest - pd.Timedelta(hours=8 * i)).timestamp() * 1000),
                   "fundingRate": str(.0001 + i / 10000000)} for i in range(1200)]
        requests = []

        def handler(request):
            requests.append(request)
            assert request.url.path == "/api/v1/contract/funding_rate/history"
            assert request.url.params["symbol"] == "BTC_USDT"
            assert request.url.params["page_size"] == "1000"
            assert not {"apikey", "signature", "request-time"} & set(request.headers)
            page = int(request.url.params["page_num"])
            rows = events[:1000] if page == 1 else events[999:]
            return httpx.Response(200, json={"success": True, "code": 0, "data": {"resultList": rows}})

        client = _client(handler)
        try:
            boundary = events[1195]["settleTime"]
            rates = await client.get_funding_history("btc-usdt", boundary)
            assert len(requests) == 2
            assert len(rates) == 1196
            assert rates.index.is_monotonic_increasing and rates.index.is_unique
            assert str(rates.index.tz) == "UTC"
            assert rates.index[0] == pd.Timestamp(boundary, unit="ms", tz="UTC")
            assert rates.index[-1] == newest
            assert rates.iloc[-1] == pytest.approx(float(events[0]["fundingRate"]))
        finally:
            await client.close()
    asyncio.run(run())


def test_empty_funding_history_preserves_utc_float_schema():
    async def run():
        client = _client(lambda request: httpx.Response(200, json={"success": True, "data": {"resultList": []}}))
        try:
            rates = await client.get_funding_history("BTC_USDT", 0)
            assert rates.empty and str(rates.index.tz) == "UTC"
            assert rates.dtype == float and rates.name == "funding_rate"
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize("rate", ["nan", "inf", "-inf"])
def test_nonfinite_public_funding_is_rejected(rate):
    async def run():
        client = _client(lambda request: httpx.Response(200, json={"success": True,
            "data": {"resultList": [{"settleTime": 1000, "fundingRate": rate}]}}))
        try:
            with pytest.raises(MEXCAPIError):
                await client.get_funding_history("BTC_USDT", 0)
        finally:
            await client.close()
    asyncio.run(run())
