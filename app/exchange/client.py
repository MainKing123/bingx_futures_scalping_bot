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

    @staticmethod
    def _extract_kline_rows(raw: dict | list) -> list:
        return raw if isinstance(raw, list) else raw.get("data", raw.get("result", []))

    @staticmethod
    def _row_timestamp_ms(row) -> int | None:
        if isinstance(row, dict):
            value = row.get("time", row.get("timestamp"))
        elif isinstance(row, (list, tuple)) and row:
            value = row[0]
        else:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _rows_to_klines_frame(rows: list) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        frame = pd.DataFrame(rows)
        if isinstance(rows[0], dict):
            ts_col = "time" if "time" in frame.columns else "timestamp"
            frame = frame.rename(columns={ts_col: "timestamp"})
            required = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required:
                if col not in frame.columns:
                    frame[col] = None
            frame = frame[required]
        else:
            width = len(rows[0]) if rows and isinstance(rows[0], (list, tuple)) else 0
            if width >= 7:
                frame = frame.iloc[:, :7]
                frame.columns = ["timestamp", "open", "close", "high", "low", "volume", "turnover"]
                frame = frame[["timestamp", "open", "high", "low", "close", "volume"]]
            elif width == 6:
                frame = frame.iloc[:, :6]
                frame.columns = ["timestamp", "open", "high", "low", "close", "volume"]
            else:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).set_index("timestamp")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame

    async def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time: int | None = None, end_time: int | None = None) -> pd.DataFrame:
        # BingX enforces max 1440 candles per request; paginate backward for larger history.
        max_limit = 1440
        requested_limit = max(1, int(limit))
        remaining = requested_limit
        cursor_end = int(end_time) if end_time is not None else None
        collected_rows: list = []

        while remaining > 0:
            chunk_limit = min(max_limit, remaining)
            params = {"symbol": symbol, "interval": interval, "limit": chunk_limit}
            if start_time is not None:
                params["startTime"] = int(start_time)
            if cursor_end is not None:
                params["endTime"] = int(cursor_end)

            raw = await self._request("GET", SWAP_KLINES, params=params, signed=False)
            rows = self._extract_kline_rows(raw)
            if not rows:
                break

            collected_rows.extend(rows)
            remaining -= len(rows)

            ts_values = [self._row_timestamp_ms(row) for row in rows]
            ts_values = [x for x in ts_values if x is not None]
            if not ts_values:
                break
            oldest_ts = min(ts_values)

            if len(rows) < chunk_limit:
                break
            next_end = oldest_ts - 1
            if start_time is not None and next_end < int(start_time):
                break
            if cursor_end is not None and next_end >= cursor_end:
                break
            cursor_end = next_end

        frame = self._rows_to_klines_frame(collected_rows)
        if frame.empty:
            return frame

        if start_time is not None:
            start_dt = pd.to_datetime(int(start_time), unit="ms", utc=True)
            frame = frame[frame.index >= start_dt]
        if end_time is not None:
            end_dt = pd.to_datetime(int(end_time), unit="ms", utc=True)
            frame = frame[frame.index <= end_dt]
        if len(frame) > requested_limit:
            frame = frame.tail(requested_limit)
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

    @staticmethod
    def _volatility_score(frame: pd.DataFrame) -> float:
        if frame.empty or len(frame) < 5:
            return 0.0
        close = pd.to_numeric(frame["close"], errors="coerce")
        spread = (pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(frame["low"], errors="coerce")).abs()
        spread_pct = (spread / close.replace(0, pd.NA) * 100).dropna()
        if spread_pct.empty:
            return 0.0
        return float(spread_pct.mean())

    async def get_top_volatile_symbols(
        self,
        limit: int = 20,
        min_volume_usd: float = 10_000_000,
        pool_size: int = 60,
        interval: str = "5m",
        lookback: int = 96,
    ) -> list[dict]:
        candidates = await self.get_top_symbols(limit=max(limit, pool_size), min_volume_usd=min_volume_usd)
        if not candidates:
            return []

        sem = asyncio.Semaphore(8)

        async def score(symbol: str) -> dict | None:
            async with sem:
                try:
                    frame = await self.get_klines(symbol, interval=interval, limit=lookback)
                    value = self._volatility_score(frame)
                    if value <= 0:
                        return None
                    return {"symbol": symbol, "volatility": round(value, 4)}
                except Exception as exc:
                    logger.warning(f"Volatility calculation failed for {symbol}: {exc}")
                    return None

        ranked = [row for row in await asyncio.gather(*(score(symbol) for symbol in candidates)) if row is not None]
        ranked.sort(key=lambda row: row["volatility"], reverse=True)
        return ranked[:limit]

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
