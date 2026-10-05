import asyncio
import gzip
import json
import time

import pytest

from app.exchange.ws_stream import MEXCKlineStream


class Socket:
    def __init__(self):
        self.sent = []
        self.closed = False
    async def send(self, value):
        self.sent.append(json.loads(value))
    async def close(self):
        self.closed = True


def kline(t, interval="Min5", **extra):
    return json.dumps({"channel": "push.kline", "data": {"symbol": "BTC_USDT", "interval": interval, "t": t, "o": 100, "c": 101, **extra}})


def test_subscription_protocol_and_symbol_wide_unsubscribe_preserves_other_interval():
    async def run():
        stream = MEXCKlineStream()
        stream.ws = Socket()
        await stream.subscribe("BTC-USDT", "5m")
        await stream.subscribe("BTC_USDT", "15m")
        await stream.subscribe("BTC-USDT", "5m")
        assert len(stream.ws.sent) == 2
        assert stream.ws.sent[0] == {"method": "sub.kline", "param": {"symbol": "BTC_USDT", "interval": "Min5"}, "gzip": False}
        await stream.unsubscribe("BTC-USDT", "5m")
        assert stream.ws.sent[-2] == {"method": "unsub.kline", "param": {"symbol": "BTC_USDT"}}
        assert stream.ws.sent[-1]["param"]["interval"] == "Min15"
        assert stream.subscriptions == {"BTC_USDT": {"15m"}}
        with pytest.raises(ValueError):
            await stream.subscribe("BTC_USDT", "3m")
        await stream.close()
    asyncio.run(run())


def test_only_later_window_emits_close_once_with_last_update_and_utc_units():
    async def run():
        stream = MEXCKlineStream()
        stream.finalization_delay = 0
        emitted = []
        async def callback(symbol, interval, candle):
            emitted.append((symbol, interval, candle))
        stream.on_kline_close(callback)
        await stream.subscribe("BTC-USDT", "5m")
        start = 1_700_000_100
        await stream._handle_message(kline(start))
        await stream._handle_message(gzip.compress(kline(start, c=102).encode()))
        assert not emitted
        await stream._handle_message(kline(start + 300))
        await stream._handle_message(kline(start + 300))
        await stream._handle_message(kline(start - 300))
        await asyncio.gather(*stream._callback_tasks)
        assert len(emitted) == 1
        symbol, interval, candle = emitted[0]
        assert (symbol, interval) == ("BTC_USDT", "5m")
        assert candle["c"] == 102 and candle["t"] == start
        assert candle["timestamp"] == start * 1000
        await stream.close()
    asyncio.run(run())


def test_reconnect_duplicate_is_not_emitted_again_and_unsubscribed_messages_ignored():
    async def run():
        stream = MEXCKlineStream()
        stream.finalization_delay = 0
        emitted = []
        async def callback(*args):
            emitted.append(args)
        stream.on_kline_close(callback)
        await stream.subscribe("BTC_USDT", "5m")
        start = 1_700_000_100
        await stream._handle_message(kline(start))
        await stream._handle_message(kline(start + 300))
        await asyncio.gather(*stream._callback_tasks)
        stream._windows.clear()
        await stream._handle_message(kline(start))
        await stream._handle_message(kline(start + 300))
        await stream._handle_message(kline(start + 900, "Min15"))
        await stream._handle_message('{"channel":"pong","data":123}')
        await stream._handle_message("invalid json")
        await stream._handle_message(b"\x1f\x8b\x08")
        await asyncio.gather(*stream._callback_tasks)
        assert len(emitted) == 1
        await stream.close()
    asyncio.run(run())


def test_missing_application_pong_closes_stream_and_pong_updates_clock():
    async def run():
        stream = MEXCKlineStream()
        stream.ws = Socket()
        stream._running = True
        stream._last_pong = time.monotonic() - 60
        await stream._heartbeat()
        assert stream.ws.closed
        await stream._handle_message('{"channel":"pong","data":123}')
        assert time.monotonic() - stream._last_pong < 1
        await stream.close()
    asyncio.run(run())


def test_close_cancels_pending_delayed_notification():
    async def run():
        stream = MEXCKlineStream()
        emitted = []
        async def callback(*args):
            emitted.append(args)
        stream.on_kline_close(callback)
        await stream.subscribe("BTC_USDT", "5m")
        await stream._handle_message(kline(1_700_000_100))
        await stream._handle_message(kline(1_700_000_400))
        await stream.close()
        assert not emitted
    asyncio.run(run())
