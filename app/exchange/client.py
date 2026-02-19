from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from urllib.parse import urlencode

import httpx
import pandas as pd
from loguru import logger

from app.config import Settings
from app.exchange.endpoints import (
    BASE_URL,
    SWAP_BALANCE,
    SWAP_CANCEL,
    SWAP_CONTRACTS,
    SWAP_DEPTH,
    SWAP_KLINES,
    SWAP_LEVERAGE,
    SWAP_OPEN_INTEREST,
    SWAP_ORDER,
    SWAP_POSITIONS,
    SWAP_PREMIUM_INDEX,
    SWAP_TICKER,
)


class BingXAPIError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(f"BingX API Error {code}: {message}")
        self.code = code
        self.message = message


class BingXClient:
    def __init__(self, settings: Settings):
        self.api_key = settings.bingx_api_key
        self.api_secret = settings.bingx_api_secret
        self.client = httpx.AsyncClient(base_url=BASE_URL, timeout=15.0, headers={"X-BX-APIKEY": self.api_key})

    def _sign(self, params: dict) -> dict:
        params = {**params, "timestamp": int(time.time() * 1000)}
        sorted_params = dict(sorted(params.items()))
        query = urlencode(sorted_params)
        signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        sorted_params["signature"] = signature
        return sorted_params

    async def _request(self, method: str, path: str, params: dict | None = None, signed: bool = True) -> dict:
        payload = params or {}
        if signed:
            payload = self._sign(payload)
        for attempt in range(4):
            try:
                response = await self.client.request(method, path, params=payload if method != "POST" else None, data=payload if method == "POST" else None)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("retry", request=response.request, response=response)
                response.raise_for_status()
                body = response.json()
                if body.get("code", 0) != 0:
                    raise BingXAPIError(body.get("code", -1), body.get("msg", "Unknown"))
                return body.get("data", body)
            except (httpx.HTTPError, BingXAPIError) as e:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                logger.warning(f"Request failed {method} {path}: {e}; retry in {wait}s")
                await asyncio.sleep(wait)
        raise RuntimeError("Unreachable")

    async def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time: int | None = None, end_time: int | None = None) -> pd.DataFrame:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        raw = await self._request("GET", SWAP_KLINES, params=params, signed=False)
        rows = raw if isinstance(raw, list) else raw.get("data", raw.get("result", []))
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame = pd.DataFrame(rows)
        frame = frame.iloc[:, :7]
        frame.columns = ["timestamp", "open", "close", "high", "low", "volume", "turnover"]
        frame = frame[["timestamp", "open", "high", "low", "close", "volume"]]
        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.set_index("timestamp").sort_index()
        return frame

    @staticmethod
    def _is_tradeable_usdt_symbol(symbol: str) -> bool:
        excluded = {"USDC-USDT", "BUSD-USDT", "DAI-USDT", "TUSD-USDT", "FDUSD-USDT"}
        return symbol.endswith("-USDT") and symbol not in excluded

    async def get_contracts(self) -> list[dict]:
        contracts = await self._request("GET", SWAP_CONTRACTS, signed=False)
        return [c for c in contracts if self._is_tradeable_usdt_symbol(c.get("symbol", ""))]

    async def get_tickers(self) -> list[dict]:
        tickers = await self._request("GET", SWAP_TICKER, signed=False)
        return [t for t in tickers if self._is_tradeable_usdt_symbol(t.get("symbol", ""))]
    async def get_ticker(self, symbol: str) -> dict: return await self._request("GET", SWAP_TICKER, {"symbol": symbol}, signed=False)

    async def get_top_symbols(self, limit: int = 30, min_volume_usd: float = 10_000_000) -> list[str]:
        tickers = await self.get_tickers()
        items = [t for t in tickers if float(t.get("quoteVolume") or 0) > min_volume_usd]
        items.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
        return [item["symbol"] for item in items[:limit]]

    async def get_depth(self, symbol: str, limit: int = 20) -> dict: return await self._request("GET", SWAP_DEPTH, {"symbol": symbol, "limit": limit}, signed=False)
    async def get_mark_price(self, symbol: str) -> dict: return await self._request("GET", SWAP_PREMIUM_INDEX, {"symbol": symbol}, signed=False)
    async def get_open_interest(self, symbol: str) -> dict: return await self._request("GET", SWAP_OPEN_INTEREST, {"symbol": symbol}, signed=False)
    async def get_balance(self) -> dict: return await self._request("GET", SWAP_BALANCE)
    async def get_positions(self, symbol: str | None = None) -> list[dict]: return await self._request("GET", SWAP_POSITIONS, {"symbol": symbol} if symbol else {})

    async def place_order(self, symbol: str, side: str, position_side: str, order_type: str, quantity: float, price: float | None = None, stop_price: float | None = None) -> dict:
        params = {"symbol": symbol, "side": side, "positionSide": position_side, "type": order_type, "quantity": quantity}
        if price is not None: params["price"] = price
        if stop_price is not None: params["stopPrice"] = stop_price
        return await self._request("POST", SWAP_ORDER, params)

    async def cancel_order(self, symbol: str, order_id: str) -> dict: return await self._request("DELETE", SWAP_CANCEL, {"symbol": symbol, "orderId": order_id})
    async def set_leverage(self, symbol: str, side: str, leverage: int) -> dict: return await self._request("POST", SWAP_LEVERAGE, {"symbol": symbol, "side": side, "leverage": leverage})
    async def close(self): await self.client.aclose()
