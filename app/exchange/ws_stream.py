from __future__ import annotations

import asyncio
import gzip
import json
from uuid import uuid4

import websockets
from loguru import logger

from app.exchange.endpoints import WS_MARKET_URL


class BingXKlineStream:
    def __init__(self):
        self.ws = None
        self.subscriptions: dict[str, set[str]] = {}
        self.callbacks: list[callable] = []
        self._running = False

    async def connect(self):
        self._running = True
        while self._running:
            try:
                self.ws = await websockets.connect(WS_MARKET_URL)
                await self._resubscribe()
                await self._listen()
            except Exception as e:
                logger.warning(f"WS reconnect after error: {e}")
                await asyncio.sleep(5)

    async def _resubscribe(self):
        for symbol, intervals in self.subscriptions.items():
            for interval in intervals:
                await self.subscribe(symbol, interval)

    async def subscribe(self, symbol: str, interval: str):
        self.subscriptions.setdefault(symbol, set()).add(interval)
        if self.ws:
            await self.ws.send(json.dumps({"id": uuid4().hex, "reqType": "sub", "dataType": f"{symbol}@kline_{interval}"}))

    async def unsubscribe(self, symbol: str, interval: str):
        self.subscriptions.get(symbol, set()).discard(interval)
        if self.ws:
            await self.ws.send(json.dumps({"id": uuid4().hex, "reqType": "unsub", "dataType": f"{symbol}@kline_{interval}"}))

    def on_kline_close(self, callback: callable):
        self.callbacks.append(callback)

    async def _listen(self):
        assert self.ws is not None
        async for message in self.ws:
            if isinstance(message, bytes):
                payload = json.loads(gzip.decompress(message).decode("utf-8"))
            else:
                payload = json.loads(message)
            if "ping" in payload:
                await self.ws.send(json.dumps({"pong": payload["ping"]}))
                continue
            data = payload.get("data", {})
            kline = data.get("K", {})
            if not kline:
                continue
            if data.get("E", 0) >= kline.get("T", 0):
                interval = str(kline.get("i", "1min")).replace("min", "m")
                for cb in self.callbacks:
                    await cb(data.get("s"), interval, kline)

    async def close(self):
        self._running = False
        if self.ws:
            await self.ws.close()
