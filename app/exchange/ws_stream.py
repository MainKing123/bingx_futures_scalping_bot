from __future__ import annotations

import asyncio
import gzip
import json
from collections.abc import Awaitable, Callable
from uuid import uuid4

import websockets
from loguru import logger

from app.exchange.endpoints import WS_MARKET_URL

INTERVAL_MAP = {
    "1min": "1m",
    "3min": "3m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "60min": "1h",
    "2hour": "2h",
    "4hour": "4h",
    "6hour": "6h",
    "8hour": "8h",
    "12hour": "12h",
    "1day": "1d",
    "3day": "3d",
    "1week": "1w",
    "1mon": "1M",
}


class BingXKlineStream:
    def __init__(self):
        self.ws = None
        self.subscriptions: dict[str, set[str]] = {}
        self.callbacks: list[Callable[[str, str, dict], Awaitable[None]]] = []
        self._running = False

    async def connect(self):
        self._running = True
        while self._running:
            try:
                self.ws = await websockets.connect(WS_MARKET_URL)
                logger.info("Connected to BingX WS")
                await self._resubscribe()
                await self._listen()
            except Exception as e:
                logger.warning(f"WS reconnect after error: {e}")
                await asyncio.sleep(5)

    async def _resubscribe(self):
        for symbol, intervals in self.subscriptions.items():
            for interval in intervals:
                await self._send_sub(symbol, interval, "sub")

    async def _send_sub(self, symbol: str, interval: str, req_type: str):
        if self.ws:
            await self.ws.send(json.dumps({"id": uuid4().hex, "reqType": req_type, "dataType": f"{symbol}@kline_{interval}"}))

    async def subscribe(self, symbol: str, interval: str):
        self.subscriptions.setdefault(symbol, set()).add(interval)
        await self._send_sub(symbol, interval, "sub")

    async def unsubscribe(self, symbol: str, interval: str):
        if symbol in self.subscriptions:
            self.subscriptions[symbol].discard(interval)
            if not self.subscriptions[symbol]:
                self.subscriptions.pop(symbol, None)
        await self._send_sub(symbol, interval, "unsub")

    def on_kline_close(self, callback: Callable[[str, str, dict], Awaitable[None]]):
        self.callbacks.append(callback)

    async def _listen(self):
        assert self.ws is not None
        async for message in self.ws:
            payload = json.loads(gzip.decompress(message).decode("utf-8")) if isinstance(message, bytes) else json.loads(message)
            if "ping" in payload:
                await self.ws.send(json.dumps({"pong": payload["ping"]}))
                continue

            data = payload.get("data", {})
            kline = data.get("K", {})
            if not kline:
                continue
            if data.get("E", 0) >= kline.get("T", 0):
                interval = INTERVAL_MAP.get(str(kline.get("i", "1min")), str(kline.get("i", "1min")))
                for cb in self.callbacks:
                    await cb(data.get("s"), interval, kline)

    async def close(self):
        self._running = False
        if self.ws:
            await self.ws.close()
