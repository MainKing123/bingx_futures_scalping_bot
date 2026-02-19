import asyncio
from datetime import datetime, timezone

from app.config import Settings
from app.exchange.client import BingXClient


def _build_rows(count: int) -> list[list]:
    start_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    step_ms = 60_000
    rows = []
    for i in range(count):
        ts = start_ms + i * step_ms
        rows.append([ts, "100", "101", "102", "99", "1000"])
    return rows


def test_get_klines_paginates_when_limit_exceeds_exchange_cap():
    async def run():
        client = BingXClient(Settings())
        all_rows = _build_rows(2600)

        async def fake_request(method, path, params=None, signed=False):
            limit = int(params.get("limit", 0))
            assert limit <= 1440
            end_time = int(params["endTime"]) if "endTime" in params else None
            if end_time is None:
                data = all_rows[-limit:]
            else:
                eligible = [row for row in all_rows if int(row[0]) <= end_time]
                data = eligible[-limit:]
            return data

        client._request = fake_request
        frame = await client.get_klines("BTC-USDT", "1m", limit=2500)
        assert len(frame) == 2500
        await client.close()

    asyncio.run(run())


def test_get_klines_single_call_for_small_limit():
    async def run():
        client = BingXClient(Settings())
        calls = {"n": 0}

        async def fake_request(method, path, params=None, signed=False):
            calls["n"] += 1
            return _build_rows(120)

        client._request = fake_request
        frame = await client.get_klines("BTC-USDT", "1m", limit=120)
        assert len(frame) == 120
        assert calls["n"] == 1
        await client.close()

    asyncio.run(run())
