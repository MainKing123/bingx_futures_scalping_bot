from __future__ import annotations

import asyncio
import gzip
import json
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress

import websockets
from loguru import logger

from app.exchange.client import normalize_symbol
from app.exchange.endpoints import INTERVALS, WS_MARKET_URL, normalize_interval


class MEXCKlineStream:
    """Market notifications; callers fetch closed REST bars for strategy decisions.

    MEXC's push.kline has no closed flag. A later window proves the prior
    window ended. The last stored update is only a notification, not a
    guaranteed final OHLC snapshot. Duplicates/out-of-order pushes are ignored.
    """

    def __init__(self):
        self.ws = None
        self.subscriptions: dict[str, set[str]] = {}
        self.callbacks: list[Callable[[str, str, dict], Awaitable[None]]] = []
        self._running = False
        self._stop = asyncio.Event()
        self._windows: dict[tuple[str, str], dict] = {}
        self._emitted: dict[tuple[str, str], int] = {}
        self._subscription_lock = asyncio.Lock()
        self._callback_tasks: set[asyncio.Task] = set()
        self.finalization_delay = 2.0
        self._last_pong = time.monotonic()

    async def connect(self):
        self._running = True
        self._stop.clear()
        while self._running:
            heartbeat = None
            try:
                async with websockets.connect(WS_MARKET_URL, ping_interval=None) as ws:
                    self.ws = ws
                    self._last_pong = time.monotonic()
                    self._windows.clear()
                    logger.info("Connected to MEXC market WebSocket")
                    await self._resubscribe()
                    heartbeat = asyncio.create_task(self._heartbeat())
                    await self._listen()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("MEXC market WebSocket disconnected; reconnecting")
            finally:
                if heartbeat:
                    heartbeat.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await heartbeat
                self.ws = None
            if self._running:
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=5)

    async def _heartbeat(self):
        while self._running and self.ws is not None:
            if time.monotonic() - self._last_pong > 45:
                await self.ws.close()
                return
            await self.ws.send(json.dumps({"method": "ping"}))
            await asyncio.sleep(15)

    async def _resubscribe(self):
        async with self._subscription_lock:
            for symbol, intervals in list(self.subscriptions.items()):
                for interval in sorted(intervals):
                    await self._send_sub(symbol, interval)

    async def _send_sub(self, symbol: str, interval: str):
        if self.ws is not None:
            await self.ws.send(json.dumps({"method": "sub.kline", "param": {
                "symbol": symbol, "interval": INTERVALS[interval][0]}, "gzip": False}))

    async def subscribe(self, symbol: str, interval: str):
        symbol, interval = normalize_symbol(symbol), normalize_interval(interval)
        async with self._subscription_lock:
            if interval in self.subscriptions.get(symbol, set()):
                return
            self.subscriptions.setdefault(symbol, set()).add(interval)
            await self._send_sub(symbol, interval)

    async def unsubscribe(self, symbol: str, interval: str):
        symbol, interval = normalize_symbol(symbol), normalize_interval(interval)
        async with self._subscription_lock:
            if interval not in self.subscriptions.get(symbol, set()):
                return
            self.subscriptions[symbol].discard(interval)
            self._windows.pop((symbol, interval), None)
            # Official unsubscribe is symbol-wide, so restore the other intervals.
            if self.ws is not None:
                await self.ws.send(json.dumps({"method": "unsub.kline", "param": {"symbol": symbol}}))
            if self.subscriptions[symbol]:
                for remaining in sorted(self.subscriptions[symbol]):
                    await self._send_sub(symbol, remaining)
            else:
                self.subscriptions.pop(symbol)

    def on_kline_close(self, callback: Callable[[str, str, dict], Awaitable[None]]):
        self.callbacks.append(callback)

    async def _handle_message(self, message: str | bytes):
        try:
            if isinstance(message, bytes):
                message = gzip.decompress(message).decode("utf-8") if message[:2] == b"\x1f\x8b" else message.decode("utf-8")
            payload = json.loads(message)
            if not isinstance(payload, dict):
                return
            if payload.get("channel") == "pong":
                self._last_pong = time.monotonic()
                return
            if payload.get("channel") != "push.kline":
                return
            data = payload.get("data")
            if not isinstance(data, dict):
                return
            symbol = normalize_symbol(data.get("symbol", payload.get("symbol", "")))
            interval = normalize_interval(data.get("interval", ""))
            start = int(data["t"])
            if start <= 0 or interval not in self.subscriptions.get(symbol, set()):
                return
        except (ValueError, TypeError, KeyError, OSError, EOFError, UnicodeError):
            logger.warning("Ignoring malformed MEXC kline message")
            return
        key = (symbol, interval)
        previous = self._windows.get(key)
        if previous and start < previous["t"]:
            return
        current = {**data, "symbol": symbol, "interval": interval, "t": start,
                   "time": start * 1000, "timestamp": start * 1000}
        self._windows[key] = current
        if not previous or start == previous["t"] or previous["t"] <= self._emitted.get(key, 0):
            return
        self._emitted[key] = previous["t"]
        task = asyncio.create_task(self._dispatch(symbol, interval, previous))
        self._callback_tasks.add(task)
        task.add_done_callback(self._callback_tasks.discard)

    async def _dispatch(self, symbol: str, interval: str, candle: dict):
        # Let the REST publication grace pass without blocking other symbols.
        await asyncio.sleep(self.finalization_delay)
        if interval not in self.subscriptions.get(symbol, set()):
            return
        for callback in self.callbacks:
            try:
                await callback(symbol, interval, candle)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("MEXC candle callback failed for {} {}", symbol, interval)

    async def _listen(self):
        assert self.ws is not None
        async for message in self.ws:
            await self._handle_message(message)

    async def close(self):
        self._running = False
        self._stop.set()
        tasks = list(self._callback_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self.ws is not None:
            await self.ws.close()
