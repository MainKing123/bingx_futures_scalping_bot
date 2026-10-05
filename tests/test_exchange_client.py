import asyncio
import hashlib
import hmac
import json
from decimal import Decimal

import httpx
import pandas as pd
import pytest

from app.config import Settings
from app.exchange.client import MEXCAPIError, MEXCClient, contracts_for_notional, round_bracket_prices
from app.exchange import endpoints as ep


def contract(symbol="BTC_USDT", **kwargs):
    return {"symbol": symbol, "contractSize": "0.001", "volUnit": "1", "volScale": 0,
            "minVol": "1", "maxVol": "100000", "priceUnit": "0.1", "priceScale": 1,
            "quoteCoin": "USDT", "settleCoin": "USDT", "apiAllowed": True, "state": 0,
            "positionOpenType": 3, "minLeverage": 1, "maxLeverage": 20, **kwargs}


def configured_client(handler):
    settings = Settings(mexc_api_key="synthetic-test-access", mexc_api_secret="synthetic-test-secret")
    client = MEXCClient(settings, transport=httpx.MockTransport(handler))
    async def no_throttle(path):
        pass
    async def clock(**kwargs):
        return 1_700_000_000_000
    client._throttle = no_throttle
    client.get_server_time = clock
    client._timestamp_ms = lambda: 1_700_000_000_000
    return client


def test_signing_matches_wire_query_and_json_and_excludes_path():
    async def run():
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"success": True, "code": 0, "data": {}})
        client = configured_client(handler)
        await client._request("GET", "/api/v1/private/example/BTC_USDT", {"z": "x,y", "a": 7, "none": None})
        await client._request("POST", ep.ORDER, {"symbol": "BTC_USDT", "price": Decimal("0.123456789123456789"), "vol": Decimal("2"), "nothing": None})
        for request in requests:
            params = request.url.query.decode() if request.method == "GET" else request.content.decode()
            signature = hmac.new(client.api_secret.encode(), (client.api_key + "1700000000000" + params).encode(), hashlib.sha256).hexdigest()
            assert request.headers["Signature"] == signature
            assert request.headers["ApiKey"] == client.api_key
            assert request.headers["Recv-Window"] == "10"
        assert requests[0].url.query.decode() == "a=7&z=x%2Cy"
        assert "0.123456789123456789" in requests[1].content.decode()
        assert "nothing" not in requests[1].content.decode()
        await client.close()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["timeout", "http500", "api500", "duplicate", "malformed"])
def test_mutations_are_never_retried_and_errors_do_not_echo_credentials(failure):
    async def run():
        calls = []
        def handler(request):
            calls.append(request)
            if failure == "timeout":
                raise httpx.ReadTimeout("synthetic-test-secret leaked raw transport", request=request)
            if failure == "http500":
                return httpx.Response(500, text="synthetic-test-secret")
            if failure == "malformed":
                return httpx.Response(200, text="synthetic-test-secret")
            return httpx.Response(200, json={"success": False, "code": 2042 if failure == "duplicate" else 500,
                                            "message": "synthetic-test-secret", "data": None})
        client = configured_client(handler)
        with pytest.raises(MEXCAPIError) as exc:
            await client._request("POST", ep.ORDER, {"externalOid": "deterministic_id"})
        assert len(calls) == 1
        assert exc.value.uncertain
        assert "synthetic-test" not in str(exc.value)
        await client.close()
    asyncio.run(run())


def test_auth_error_is_not_retried_and_public_requests_carry_no_key():
    async def run():
        calls = []
        def handler(request):
            calls.append(request)
            if request.url.path == ep.TICKER:
                assert "ApiKey" not in request.headers
                return httpx.Response(200, json={"success": True, "data": []})
            return httpx.Response(200, json={"success": False, "code": 602, "message": "secret"})
        client = configured_client(handler)
        await client._request("GET", ep.TICKER, signed=False)
        with pytest.raises(MEXCAPIError) as exc:
            await client.get_balance()
        assert exc.value.code == 602
        assert len(calls) == 2
        await client.close()
    asyncio.run(run())


def test_contract_sizing_rounds_down_without_promoting_small_orders():
    assert contracts_for_notional(contract(), "12.345", "100") == Decimal("123")
    stepped = contract(contractSize="0.01", volUnit="0.25", volScale=2, minVol="0.25")
    assert contracts_for_notional(stepped, "10.37", "100") == Decimal("10.25")
    with pytest.raises(ValueError, match="minimum"):
        contracts_for_notional(contract(), "0.01", "100")
    with pytest.raises(ValueError):
        contracts_for_notional(contract(), "NaN", "100")
    with pytest.raises(ValueError, match="maximum"):
        contracts_for_notional(contract(maxVol=1), "10", "100")


@pytest.mark.parametrize("direction,side,entry,stop,take", [
    ("LONG", 1, "100.0", "99.0", "102.0"),
    ("SHORT", 3, "100.1", "101.1", "98.1"),
])
def test_limit_bracket_uses_contracts_attached_exits_and_rounded_price(direction, side, entry, stop, take):
    async def run():
        payloads = []
        def handler(request):
            if request.url.path == ep.CONTRACTS:
                assert request.url.params["symbol"] == "BTC_USDT"
                data = contract()
            elif request.url.path == ep.POSITION_MODE:
                data = {"positionMode": 1}
            elif request.url.path == ep.ORDER:
                payloads.append(json.loads(request.content, parse_float=Decimal))
                data = {"orderId": "12345", "ts": 1_700_000_000_000}
            else:
                raise AssertionError(request.url.path)
            return httpx.Response(200, json={"success": True, "code": 0, "data": data})
        client = configured_client(handler)
        result = await client.place_bracket_order("BTC-USDT", direction, "12.345", "100.05",
                                                  "99.01" if direction == "LONG" else "101.09",
                                                  "102.09" if direction == "LONG" else "98.01", 2, "signal_hash_01")
        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["type"] == 1 and payload["openType"] == 1
        assert payload["side"] == side and payload["symbol"] == "BTC_USDT"
        assert Decimal(str(payload["price"])) == Decimal(entry)
        assert Decimal(str(payload["stopLossPrice"])) == Decimal(stop)
        assert Decimal(str(payload["takeProfitPrice"])) == Decimal(take)
        assert payload["vol"] == 123
        assert result["orderId"] == "12345" and result["contractSize"] == "0.001"
        assert result["vol"] == "123"
        assert Decimal(result["riskReward"]) >= 2
        assert Decimal(result["riskUsdt"]) <= Decimal("12.345") * abs(Decimal("100.05") - Decimal("99.01" if direction == "LONG" else "101.09")) / Decimal("100.05")
        await client.close()
    asyncio.run(run())


@pytest.mark.parametrize("changed", [{"apiAllowed": False}, {"state": 4}, {"positionOpenType": 2}])
def test_bracket_rejects_untradeable_contract_before_mutation(changed):
    async def run():
        def handler(request):
            assert request.method == "GET"
            return httpx.Response(200, json={"success": True, "data": contract(**changed)})
        client = configured_client(handler)
        with pytest.raises(ValueError):
            await client.place_bracket_order("BTC_USDT", "LONG", 10, 100, 99, 102, 2, "signal01")
        await client.close()
    asyncio.run(run())


def _candles(times):
    return {"time": times, "open": [100] * len(times), "high": [102] * len(times),
            "low": [99] * len(times), "close": [101] * len(times), "vol": [1000] * len(times)}


def test_get_klines_paginates_seconds_excludes_unclosed_bar_and_normalizes_volume():
    async def run():
        base = 1_700_000_040  # aligned to a minute
        times = [base + i * 60 for i in range(4301)]
        calls = []
        client = MEXCClient(Settings())
        async def fake_request(method, path, params=None, signed=False):
            if path == ep.CONTRACTS:
                return contract()
            assert path == ep.KLINES.format(symbol="BTC_USDT")
            assert params["interval"] == "Min1"
            assert params["end"] < 10_000_000_000
            calls.append(params)
            eligible = [t for t in times if t <= params["end"]][-2000:]
            return _candles(list(reversed(eligible)))
        async def clock(**kwargs):
            return (times[-1] + 20) * 1000
        client._request = fake_request
        client.get_server_time = clock
        frame = await client.get_klines("BTC-USDT", "1m", limit=4100)
        assert len(frame) == 4100 and len(calls) == 3
        assert frame.index.is_monotonic_increasing and frame.index.is_unique
        assert str(frame.index.tz) == "UTC"
        assert frame.index[-1] == pd.Timestamp(times[-2], unit="s", tz="UTC")
        assert frame.iloc[-1]["volume"] == 1.0
        await client.close()
    asyncio.run(run())


def test_candle_range_in_ms_and_month_close_uses_calendar():
    async def run():
        dates = [pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC"), pd.Timestamp("2025-03-01", tz="UTC")]
        client = MEXCClient(Settings())
        async def fake_request(method, path, params=None, signed=False):
            return contract() if path == ep.CONTRACTS else _candles([int(d.timestamp()) for d in dates])
        async def clock(**kwargs):
            return int(pd.Timestamp("2025-03-15", tz="UTC").timestamp() * 1000)
        client._request = fake_request
        client.get_server_time = clock
        frame = await client.get_klines("BTC_USDT", "Month1", start_time=int(dates[1].timestamp() * 1000), limit=20)
        assert list(frame.index) == [dates[1]]
        with pytest.raises(ValueError, match="Unsupported"):
            await client.get_klines("BTC_USDT", "3m")
        await client.close()
    asyncio.run(run())


def test_historical_end_time_excludes_candle_closing_after_frozen_cutoff():
    async def run():
        base = int(pd.Timestamp("2026-10-04T15:20:00Z").timestamp())
        client = MEXCClient(Settings())
        async def fake_request(method, path, params=None, signed=False):
            return contract() if path == ep.CONTRACTS else _candles([base - 300, base, base + 300])
        async def clock(**kwargs):
            return int(pd.Timestamp("2026-10-04T17:00:00Z").timestamp() * 1000)
        client._request = fake_request
        client.get_server_time = clock
        cutoff = int(pd.Timestamp("2026-10-04T15:23:06.405Z").timestamp() * 1000)
        frame = await client.get_klines("BTC_USDT", "5m", end_time=cutoff)
        assert frame.index[-1] == pd.Timestamp("2026-10-04T15:15:00Z")
        assert (frame.index + pd.Timedelta(minutes=5) <= pd.Timestamp(cutoff, unit="ms", tz="UTC")).all()
        exact_close = await client.get_klines("BTC_USDT", "5m", end_time=(base + 300) * 1000)
        assert exact_close.index[-1] == pd.Timestamp("2026-10-04T15:20:00Z")
        await client.close()
    asyncio.run(run())


def test_malformed_parallel_candles_fail_closed():
    with pytest.raises(MEXCAPIError):
        MEXCClient._klines_frame({"time": [1], "open": []}, Decimal("1"))


@pytest.mark.parametrize("changed", [{"close": [float("inf")]}, {"vol": [-1]}, {"high": [99]}, {"time": [None]}])
def test_invalid_market_candles_are_rejected(changed):
    with pytest.raises(MEXCAPIError):
        MEXCClient._klines_frame({**_candles([1_700_000_100]), **changed}, Decimal("1"))


def test_tick_conversion_preserves_sweep_stop_and_minimum_rr():
    entry, stop, take = round_bracket_prices(contract(), "LONG", "100.05", "99.01", "102.09")
    assert stop <= Decimal("99.01")
    assert abs(take - entry) / abs(entry - stop) >= 2
    entry, stop, take = round_bracket_prices(contract(), "SHORT", "100.05", "101.09", "98.01")
    assert stop >= Decimal("101.09")
    assert abs(take - entry) / abs(entry - stop) >= 2
    with pytest.raises(ValueError, match="more than"):
        round_bracket_prices(contract(), "LONG", 100, 90, 130)
    with pytest.raises(ValueError, match="rounding"):
        round_bracket_prices(contract(priceUnit="10", priceScale=0), "LONG", 100, 99, 102)


def test_outward_tick_rounding_reduces_volume_to_preserve_original_cash_risk():
    async def run():
        def handler(request):
            if request.url.path == ep.CONTRACTS:
                data = contract()
            elif request.url.path == ep.POSITION_MODE:
                data = 1
            else:
                data = {"orderId": "123"}
            return httpx.Response(200, json={"success": True, "data": data})
        client = configured_client(handler)
        result = await client.place_bracket_order("BTC_USDT", "LONG", 100, "100.005", "99.01", 102, 2, "risk_test")
        original_risk = Decimal("100") * Decimal("0.995") / Decimal("100.005")
        assert Decimal(result["riskUsdt"]) <= original_risk
        assert Decimal(result["notionalUsdt"]) <= Decimal("100")
        assert Decimal(result["vol"]) < 1000
        await client.close()
    asyncio.run(run())


def test_ticker_uses_amount24_and_only_api_allowed_metadata():
    async def run():
        def handler(request):
            data = [contract(), contract("ETH_USDT", apiAllowed=False)] if request.url.path == ep.CONTRACTS else [
                {"symbol": "BTC_USDT", "amount24": 20_000_000, "volume24": 2000, "lastPrice": 100},
                {"symbol": "ETH_USDT", "amount24": 30_000_000, "volume24": 2000, "lastPrice": 100},
            ]
            return httpx.Response(200, json={"success": True, "data": data})
        client = configured_client(handler)
        tickers = await client.get_tickers()
        assert len(tickers) == 1 and tickers[0]["symbol"] == "BTC_USDT"
        assert tickers[0]["quoteVolume"] == 20_000_000 and tickers[0]["volume"] == 2
        await client.close()
    asyncio.run(run())


def test_reconciliation_paths_and_history_page_shape():
    async def run():
        requests = []
        def handler(request):
            requests.append(request)
            if request.url.path == ep.HISTORY_POSITIONS:
                data = {"resultList": [{"positionId": 7, "symbol": "BTC_USDT", "realised": "2.1"}]}
            elif request.url.path in {ep.POSITIONS, ep.STOP_ORDERS}:
                data = []
            elif request.url.path == ep.BALANCE.format(currency="USDT"):
                data = {"equity": "100", "availableBalance": "75", "currency": "USDT"}
            else:
                data = {"orderId": "123", "dealVol": 1}
            return httpx.Response(200, json={"success": True, "data": data})
        client = configured_client(handler)
        assert (await client.get_balance())["equity"] == "100"
        await client.get_order_by_external_id("BTC-USDT", "order01")
        await client.get_order("123")
        await client.get_positions("BTC-USDT")
        await client.get_stop_orders("BTC-USDT")
        result = await client.get_history_positions("BTC-USDT", page_num=2)
        assert result[0]["positionId"] == 7
        assert requests[-1].url.params["page_num"] == "2"
        assert requests[-1].url.params["symbol"] == "BTC_USDT"
        await client.close()
    asyncio.run(run())


@pytest.mark.parametrize("position_type,close_side,mode", [(1, 4, 1), (2, 2, 2)])
def test_close_position_uses_actual_available_contracts(position_type, close_side, mode):
    async def run():
        payloads = []
        def handler(request):
            if request.url.path == ep.CONTRACTS:
                data = contract()
            elif request.url.path == ep.POSITION_MODE:
                data = mode
            elif request.url.path == ep.TICKER:
                data = {"symbol": "BTC_USDT", "lastPrice": 100, "volume24": 0, "amount24": 0}
            else:
                assert request.url.path == ep.ORDER
                payloads.append(json.loads(request.content))
                data = {"orderId": "123"}
            return httpx.Response(200, json={"success": True, "data": data})
        client = configured_client(handler)
        await client.close_position({"symbol": "BTC_USDT", "positionId": 7, "positionType": position_type,
                                     "state": 1, "holdVol": 20, "frozenVol": 3, "openType": 1}, "close01")
        assert payloads[0]["vol"] == 17 and payloads[0]["side"] == close_side
        assert payloads[0]["type"] == 5
        assert payloads[0].get("reduceOnly", False) == (mode == 2)
        await client.close()
    asyncio.run(run())


def test_cancel_checks_per_item_failure_and_numeric_array_body():
    async def run():
        def handler(request):
            assert request.url.path == ep.CANCEL
            assert json.loads(request.content) == [123]
            return httpx.Response(200, json={"success": True, "data": [{"orderId": "123", "errorCode": 2041}]})
        client = configured_client(handler)
        with pytest.raises(MEXCAPIError) as exc:
            await client.cancel_order("BTC_USDT", "123")
        assert exc.value.code == 2041
        await client.close()
    asyncio.run(run())
