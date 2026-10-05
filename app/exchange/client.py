from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import math
import re
import time
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from urllib.parse import urlencode

import httpx
import pandas as pd
from loguru import logger

from app.config import Settings
from app.exchange import endpoints as ep


def normalize_symbol(symbol: str) -> str:
    result = str(symbol).strip().upper().replace("-", "_").replace("/", "_")
    if not re.fullmatch(r"[A-Z0-9]+_[A-Z0-9]+", result):
        raise ValueError("Invalid MEXC contract symbol")
    return result


def _decimal(value, field: str, *, positive: bool = True) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"Invalid {field}") from None
    if not result.is_finite() or (positive and result <= 0):
        raise ValueError(f"Invalid {field}")
    return result


def _round_step(value: Decimal, step: Decimal, rounding=ROUND_FLOOR) -> Decimal:
    return (value / step).to_integral_value(rounding=rounding) * step


def contracts_for_notional(contract: dict, notional_usdt, price) -> Decimal:
    """USDT linear futures quantity in contracts, rounded down to lot step."""
    if contract.get("quoteCoin", "USDT") != "USDT" or contract.get("settleCoin", "USDT") != "USDT":
        raise ValueError("Only USDT linear contracts are supported")
    size = _decimal(contract.get("contractSize"), "contract size")
    step = _decimal(contract.get("volUnit"), "volume step")
    amount = _decimal(notional_usdt, "notional")
    entry = _decimal(price, "entry price")
    volume = _round_step(amount / (size * entry), step)
    if volume < _decimal(contract.get("minVol"), "minimum volume"):
        raise ValueError("Notional is below the contract minimum order size")
    maximum = _decimal(contract.get("maxVol"), "maximum volume")
    if contract.get("limitMaxVol") is not None:
        maximum = min(maximum, _decimal(contract["limitMaxVol"], "maximum limit volume"))
    if volume > maximum:
        raise ValueError("Notional exceeds the contract maximum order size")
    vol_scale = int(contract.get("volScale", max(0, -step.as_tuple().exponent)))
    if volume != volume.quantize(Decimal(1).scaleb(-vol_scale)):
        raise ValueError("Contract volume step conflicts with its precision")
    return volume


def round_bracket_prices(contract: dict, direction: str, entry, stop_loss, take_profit) -> tuple[Decimal, Decimal, Decimal]:
    """Preserve the sweep stop and 2R after conversion to exchange ticks."""
    direction = str(direction).upper()
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("Direction must be LONG or SHORT")
    original_entry = _decimal(entry, "entry")
    original_stop = _decimal(stop_loss, "stop loss")
    original_take = _decimal(take_profit, "take profit")
    long = direction == "LONG"
    if not (original_stop < original_entry < original_take if long else original_take < original_entry < original_stop):
        raise ValueError("Invalid source bracket prices")
    tick = _decimal(contract["priceUnit"], "price tick")
    stop = _round_step(original_stop, tick, ROUND_FLOOR if long else ROUND_CEILING)
    take = _round_step(original_take, tick, ROUND_FLOOR if long else ROUND_CEILING)
    # Fixed target and sweep stop determine a two-to-one entry.
    theoretical_entry = (take + Decimal(2) * stop) / Decimal(3)
    rounded_entry = _round_step(theoretical_entry, tick, ROUND_FLOOR if long else ROUND_CEILING)
    if not (0 < stop < rounded_entry < take if long else 0 < take < rounded_entry < stop):
        raise ValueError("Invalid bracket prices after tick rounding")
    if abs(rounded_entry - original_entry) / original_entry > Decimal("0.01"):
        raise ValueError("Exchange ticks move the strategy entry by more than 1 percent")
    rr = abs(take - rounded_entry) / abs(rounded_entry - stop)
    if rr < 2:
        raise ValueError("Exchange ticks cannot preserve minimum 2R")
    price_scale = int(contract.get("priceScale", max(0, -tick.as_tuple().exponent)))
    precision = Decimal(1).scaleb(-price_scale)
    if any(price != price.quantize(precision) for price in (rounded_entry, stop, take)):
        raise ValueError("Contract price tick conflicts with its precision")
    return rounded_entry, stop, take


_ERROR_MESSAGES = {
    401: "Authentication failed", 402: "API key expired", 406: "IP not allowed",
    500: "Exchange internal error", 501: "Exchange busy", 510: "Rate limit exceeded",
    511: "Endpoint permission denied", 600: "Invalid request parameters",
    602: "Signature verification failed", 603: "Repeated request; reconcile first",
    604: "Endpoint under maintenance", 701: "Read permission required",
    702: "Write permission required", 703: "Trading read permission required",
    704: "Trading write permission required", 2005: "Insufficient margin",
    2030: "External order ID too long", 2040: "Order not found",
    2041: "Order cannot be cancelled", 2042: "Duplicate external order ID",
}


class MEXCAPIError(Exception):
    """Errors never retain raw responses, signed headers, URLs, or secrets."""

    def __init__(self, code: int, message: str | None = None, *, uncertain: bool = False):
        self.code = int(code)
        self.message = _ERROR_MESSAGES.get(self.code, "Exchange request failed")
        self.uncertain = bool(uncertain)
        super().__init__(f"MEXC API error {self.code}: {self.message}")


def _json_payload(value) -> str:
    """Emit Decimal as a JSON number without passing through binary float."""
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Non-finite JSON number")
        return format(value, "f")
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(str(k)) + ":" + _json_payload(v) for k, v in value.items() if v is not None) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_json_payload(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class MEXCClient:
    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None):
        self.api_key = settings.mexc_api_key
        self.api_secret = settings.mexc_api_secret
        self.receive_window = int(settings.mexc_receive_window)
        if not 1 <= self.receive_window <= 60:
            raise ValueError("MEXC receive window must be in seconds, from 1 to 60")
        self.client = httpx.AsyncClient(base_url=settings.mexc_base_url, timeout=15.0, transport=transport)
        self._clock_offset_ms = 0
        self._clock_synced_at: float | None = None
        self._clock_lock = asyncio.Lock()
        self._rate_lock = asyncio.Lock()
        self._last_request_at = 0.0
        self._last_create_at = 0.0
        self._last_leverage_at = 0.0
        self._contracts: dict[str, dict] = {}
        self._contract_times: dict[str, float] = {}
        self._contracts_updated_at: float | None = None

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000 + self._clock_offset_ms)

    def _sign(self, timestamp: str | int, parameter_string: str) -> str:
        target = self.api_key + str(timestamp) + parameter_string
        return hmac.new(self.api_secret.encode(), target.encode(), hashlib.sha256).hexdigest()

    async def _throttle(self, path: str):
        # Shared limits cover concurrent scanners. Order creation has its stricter cap.
        async with self._rate_lock:
            now = time.monotonic()
            delay = self._last_request_at + 0.21 - now
            if path == ep.ORDER:
                delay = max(delay, self._last_create_at + 0.51 - now)
            if path == ep.LEVERAGE:
                delay = max(delay, self._last_leverage_at + 1.01 - now)
            if delay > 0:
                await asyncio.sleep(delay)
            now = time.monotonic()
            self._last_request_at = now
            if path == ep.ORDER:
                self._last_create_at = now
            if path == ep.LEVERAGE:
                self._last_leverage_at = now

    async def get_server_time(self, *, refresh: bool = False) -> int:
        async with self._clock_lock:
            if refresh or self._clock_synced_at is None or time.monotonic() - self._clock_synced_at >= 60:
                before = time.time() * 1000
                server_ms = await self._request("GET", ep.SERVER_TIME, signed=False)
                after = time.time() * 1000
                if not isinstance(server_ms, (int, float)) or server_ms <= 0:
                    raise MEXCAPIError(-2)
                self._clock_offset_ms = int(server_ms - (before + after) / 2)
                self._clock_synced_at = time.monotonic()
            return self._timestamp_ms()

    async def _request(self, method: str, path: str, params=None, signed: bool = True):
        method = method.upper()
        mutation = method not in {"GET", "HEAD"}
        payload = {} if params is None else params
        if isinstance(payload, dict):
            payload = {k: v for k, v in payload.items() if v is not None}
        if signed:
            if not self.api_key or not self.api_secret:
                raise MEXCAPIError(401)
            await self.get_server_time()
        attempts = 1 if mutation else 3
        for attempt in range(attempts):
            await self._throttle(path)
            headers = {"Language": "English"}
            query = ""
            body = None
            if method in {"GET", "DELETE", "HEAD"}:
                query = urlencode(sorted(payload.items()))
                parameter_string = query
            else:
                body = _json_payload(payload)
                parameter_string = body
                headers["Content-Type"] = "application/json"
            if signed:
                timestamp = str(self._timestamp_ms())
                headers.update({"ApiKey": self.api_key, "Request-Time": timestamp,
                                "Signature": self._sign(timestamp, parameter_string),
                                "Recv-Window": str(self.receive_window)})
            try:
                response = await self.client.request(method, path + ("?" + query if query else ""),
                                                     headers=headers, content=body)
                if response.status_code >= 400:
                    error = MEXCAPIError(response.status_code, uncertain=mutation and response.status_code >= 500)
                    retryable = response.status_code in {429, 500, 502, 503, 504}
                else:
                    try:
                        result = response.json()
                    except (ValueError, TypeError):
                        raise MEXCAPIError(-2, uncertain=mutation) from None
                    if not isinstance(result, dict):
                        raise MEXCAPIError(-2, uncertain=mutation)
                    if result.get("success") is True and result.get("code", 0) == 0:
                        return result.get("data", {})
                    try:
                        code = int(result.get("code", -1))
                    except (ValueError, TypeError):
                        code = -1
                    error = MEXCAPIError(code, uncertain=mutation and code in {-1, 500, 501, 603, 2042})
                    retryable = code in {500, 501, 510}
            except httpx.HTTPError:
                error = MEXCAPIError(-1, uncertain=mutation)
                retryable = True
            if mutation or not retryable or attempt + 1 >= attempts:
                raise error from None
            logger.warning("MEXC read request retry after error code {}", error.code)
            await asyncio.sleep(0.5 * (2 ** attempt))
        raise RuntimeError("Unreachable")

    @staticmethod
    def _is_tradeable_usdt_symbol(symbol: str) -> bool:
        symbol = normalize_symbol(symbol)
        return symbol.endswith("_USDT") and symbol not in {"USDC_USDT", "BUSD_USDT", "DAI_USDT", "TUSD_USDT", "FDUSD_USDT"}

    @staticmethod
    def _items(data) -> list[dict]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("resultList"), list):
            return data["resultList"]
        if isinstance(data, dict) and "symbol" in data:
            return [data]
        raise MEXCAPIError(-2)

    async def get_contracts(self) -> list[dict]:
        raw = self._items(await self._request("GET", ep.CONTRACTS, signed=False))
        all_contracts = {}
        for contract in raw:
            symbol = normalize_symbol(contract.get("symbol", ""))
            all_contracts[symbol] = {**contract, "symbol": symbol}
        self._contracts = all_contracts
        self._contracts_updated_at = time.monotonic()
        self._contract_times = {s: self._contracts_updated_at for s in all_contracts}
        return [c for c in all_contracts.values() if self._is_tradeable_usdt_symbol(c["symbol"])
                and c.get("apiAllowed") is True and c.get("state") == 0
                and c.get("quoteCoin", "USDT") == "USDT" and c.get("settleCoin", "USDT") == "USDT"]

    async def get_contract(self, symbol: str, *, refresh: bool = False) -> dict:
        symbol = normalize_symbol(symbol)
        if refresh or symbol not in self._contracts or time.monotonic() - self._contract_times.get(symbol, 0) > 300:
            raw = self._items(await self._request("GET", ep.CONTRACTS, {"symbol": symbol}, signed=False))
            if len(raw) != 1 or normalize_symbol(raw[0].get("symbol", "")) != symbol:
                raise MEXCAPIError(-2)
            self._contracts[symbol] = {**raw[0], "symbol": symbol}
            self._contract_times[symbol] = time.monotonic()
        return self._contracts[symbol]

    def _normalize_ticker(self, ticker: dict) -> dict:
        symbol = normalize_symbol(ticker["symbol"])
        contract = self._contracts.get(symbol, {})
        size = _decimal(contract.get("contractSize"), "contract size")
        return {**ticker, "symbol": symbol, "quoteVolume": float(ticker.get("amount24") or 0),
                "volume": float(_decimal(ticker.get("volume24", 0), "volume", positive=False) * size),
                "priceChangePercent": float(ticker.get("riseFallRate") or 0) * 100}

    async def get_tickers(self) -> list[dict]:
        if self._contracts_updated_at is None or time.monotonic() - self._contracts_updated_at > 300:
            await self.get_contracts()
        tickers = self._items(await self._request("GET", ep.TICKER, signed=False))
        allowed = {s for s, c in self._contracts.items() if self._is_tradeable_usdt_symbol(s)
                   and c.get("apiAllowed") is True and c.get("state") == 0
                   and c.get("quoteCoin", "USDT") == "USDT" and c.get("settleCoin", "USDT") == "USDT"}
        return [self._normalize_ticker(t) for t in tickers if normalize_symbol(t.get("symbol", "")) in allowed]

    async def get_ticker(self, symbol: str) -> dict:
        symbol = normalize_symbol(symbol)
        await self.get_contract(symbol)
        items = self._items(await self._request("GET", ep.TICKER, {"symbol": symbol}, signed=False))
        if len(items) != 1 or normalize_symbol(items[0]["symbol"]) != symbol:
            raise MEXCAPIError(-2)
        return self._normalize_ticker(items[0])

    @staticmethod
    def _klines_frame(raw: dict, contract_size: Decimal) -> pd.DataFrame:
        if not isinstance(raw, dict):
            raise MEXCAPIError(-2)
        fields = {"time": "timestamp", "open": "open", "high": "high", "low": "low", "close": "close", "vol": "volume"}
        values = [raw.get(field) for field in fields]
        if any(not isinstance(value, list) for value in values) or len({len(value) for value in values}) != 1:
            raise MEXCAPIError(-2)
        frame = pd.DataFrame({dest: raw[source] for source, dest in fields.items()})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True, errors="coerce")
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame.isna().any().any() or any(not frame[column].map(math.isfinite).all() for column in ("open", "high", "low", "close", "volume")):
            raise MEXCAPIError(-2)
        if any(frame[column].le(0).any() for column in ("open", "high", "low", "close")) or frame["volume"].lt(0).any():
            raise MEXCAPIError(-2)
        if frame["high"].lt(frame[["open", "low", "close"]].max(axis=1)).any() or frame["low"].gt(frame[["open", "high", "close"]].min(axis=1)).any():
            raise MEXCAPIError(-2)
        frame = frame.set_index("timestamp")
        frame["volume"] = frame["volume"] * float(contract_size)
        return frame[~frame.index.duplicated(keep="last")].sort_index()

    async def get_klines(self, symbol: str, interval: str, limit: int = 500,
                         start_time: int | None = None, end_time: int | None = None) -> pd.DataFrame:
        symbol = normalize_symbol(symbol)
        interval = ep.normalize_interval(interval)
        requested = int(limit)
        if not 1 <= requested <= 100_000:
            raise ValueError("Candle limit must be from 1 to 100000")
        if start_time is not None and end_time is not None and int(start_time) > int(end_time):
            raise ValueError("Candle start must precede end")
        contract = await self.get_contract(symbol)
        size = _decimal(contract.get("contractSize"), "contract size")
        # MEXC has no close flag; server time plus publication grace excludes the active bar.
        closed_at_ms = await self.get_server_time() - 1500
        if end_time is not None:
            # Historical end_time is a closed-data cutoff, not an opening-time bound.
            closed_at_ms = min(closed_at_ms, int(end_time))
        cursor_end = closed_at_ms // 1000
        collected: list[pd.DataFrame] = []
        while True:
            raw = await self._request("GET", ep.KLINES.format(symbol=symbol),
                                      {"interval": ep.INTERVALS[interval][0], "end": cursor_end}, signed=False)
            frame = self._klines_frame(raw, size)
            if frame.empty:
                break
            oldest = int(frame.index[0].timestamp())
            close_times = frame.index + (pd.DateOffset(months=1) if interval == "1M" else pd.Timedelta(seconds=ep.INTERVALS[interval][1]))
            frame = frame[close_times <= pd.Timestamp(closed_at_ms, unit="ms", tz="UTC")]
            if start_time is not None:
                frame = frame[frame.index >= pd.Timestamp(int(start_time), unit="ms", tz="UTC")]
            if end_time is not None:
                frame = frame[frame.index <= pd.Timestamp(int(end_time), unit="ms", tz="UTC")]
            collected.append(frame)
            merged = pd.concat(collected).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            if len(merged) >= requested or len(raw["time"]) < 2000 or (start_time is not None and oldest * 1000 <= int(start_time)):
                break
            if oldest - 1 >= cursor_end:
                raise MEXCAPIError(-2)
            cursor_end = oldest - 1
        if not collected:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
        result = pd.concat(collected).sort_index()
        return result[~result.index.duplicated(keep="last")].tail(requested)

    async def get_top_symbols(self, limit: int = 30, min_volume_usd: float = 10_000_000) -> list[str]:
        tickers = await self.get_tickers()
        items = [t for t in tickers if float(t["quoteVolume"]) > min_volume_usd]
        items.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
        return [item["symbol"] for item in items[:limit]]

    @staticmethod
    def _volatility_score(frame: pd.DataFrame) -> float:
        if frame.empty or len(frame) < 5:
            return 0.0
        close = pd.to_numeric(frame["close"], errors="coerce")
        spread = (pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(frame["low"], errors="coerce")).abs()
        scores = (spread / close.where(close != 0) * 100).dropna()
        return float(scores.mean()) if not scores.empty else 0.0

    async def get_top_volatile_symbols(self, limit: int = 20, min_volume_usd: float = 10_000_000,
                                       pool_size: int = 60, interval: str = "5m", lookback: int = 96) -> list[dict]:
        candidates = await self.get_top_symbols(max(limit, pool_size), min_volume_usd)
        semaphore = asyncio.Semaphore(5)
        async def score(symbol):
            async with semaphore:
                try:
                    value = self._volatility_score(await self.get_klines(symbol, interval, lookback))
                    return {"symbol": symbol, "volatility": round(value, 4)} if value > 0 else None
                except (MEXCAPIError, ValueError):
                    logger.warning("MEXC volatility data unavailable for {}", symbol)
                    return None
        results = [r for r in await asyncio.gather(*(score(s) for s in candidates)) if r]
        return sorted(results, key=lambda r: r["volatility"], reverse=True)[:limit]

    async def get_depth(self, symbol: str, limit: int = 20) -> dict:
        symbol = normalize_symbol(symbol)
        contract = await self.get_contract(symbol)
        raw = await self._request("GET", ep.DEPTH.format(symbol=symbol), {"limit": limit}, signed=False)
        size = _decimal(contract["contractSize"], "contract size")
        return {**raw, "symbol": symbol, **{side: [[row[0], float(_decimal(row[1], "depth volume", positive=False) * size)]
                                                  for row in raw.get(side, [])] for side in ("asks", "bids")}}

    async def get_mark_price(self, symbol: str) -> dict:
        symbol = normalize_symbol(symbol)
        raw = await self._request("GET", ep.FAIR_PRICE.format(symbol=symbol), signed=False)
        return {**raw, "symbol": symbol, "markPrice": raw["fairPrice"]}

    async def get_open_interest(self, symbol: str) -> dict:
        ticker = await self.get_ticker(symbol)
        size = _decimal((await self.get_contract(symbol))["contractSize"], "contract size")
        return {"symbol": ticker["symbol"], "openInterest": float(_decimal(ticker.get("holdVol", 0), "open interest", positive=False) * size)}

    async def get_balance(self) -> dict:
        return await self._request("GET", ep.BALANCE.format(currency="USDT"))

    async def get_funding_history(self, symbol: str, since_ms: int) -> pd.Series:
        """Public settlement rates, newest-first pages normalized to ascending UTC."""
        values = {}
        for page in range(1, 101):
            raw = await self._request("GET", "/api/v1/contract/funding_rate/history",
                {"symbol": normalize_symbol(symbol), "page_num": page, "page_size": 1000}, signed=False)
            rows = self._items(raw)
            if not rows:
                break
            oldest = min(int(row["settleTime"]) for row in rows)
            for row in rows:
                timestamp = int(row["settleTime"])
                if timestamp >= since_ms:
                    values[pd.Timestamp(timestamp, unit="ms", tz="UTC")] = float(row["fundingRate"])
            if oldest <= since_ms or len(rows) < 1000:
                break
        else:
            raise ValueError("Funding history pagination exceeded safety limit")
        if not values:
            return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name="funding_rate")
        rates = pd.Series(values, dtype=float, name="funding_rate").sort_index()
        if not all(math.isfinite(value) for value in rates):
            raise MEXCAPIError(-2)
        return rates

    async def get_positions(self, symbol: str | None = None) -> list[dict]:
        return self._items(await self._request("GET", ep.POSITIONS, {"symbol": normalize_symbol(symbol)} if symbol else {}))

    async def get_history_positions(self, symbol: str | None = None, page_num: int = 1, page_size: int = 100) -> list[dict]:
        if page_num < 1 or not 1 <= page_size <= 100:
            raise ValueError("Invalid history pagination")
        params = {"page_num": page_num, "page_size": page_size}
        if symbol:
            params["symbol"] = normalize_symbol(symbol)
        return self._items(await self._request("GET", ep.HISTORY_POSITIONS, params))

    async def get_order_by_external_id(self, symbol: str, external_oid: str) -> dict:
        self._validate_external_id(external_oid)
        return await self._request("GET", ep.ORDER_BY_EXTERNAL_ID.format(symbol=normalize_symbol(symbol), external_oid=external_oid))

    async def get_order(self, order_id: str) -> dict:
        if not str(order_id).isdigit():
            raise ValueError("Invalid order ID")
        return await self._request("GET", ep.ORDER_BY_ID.format(order_id=order_id))

    async def get_stop_orders(self, symbol: str | None = None) -> list[dict]:
        return self._items(await self._request("GET", ep.STOP_ORDERS, {"symbol": normalize_symbol(symbol)} if symbol else {}))

    async def get_open_orders(self, page_num: int = 1, page_size: int = 100) -> list[dict]:
        if page_num < 1 or not 1 <= page_size <= 100:
            raise ValueError("Invalid order pagination")
        return self._items(await self._request("GET", ep.OPEN_ORDERS, {"page_num": page_num, "page_size": page_size}))

    async def get_position_mode(self) -> int:
        data = await self._request("GET", ep.POSITION_MODE)
        value = data.get("positionMode") if isinstance(data, dict) else data
        if isinstance(value, bool) or value not in (1, 2):
            raise MEXCAPIError(-2)
        return int(value)

    @staticmethod
    def _validate_external_id(external_oid: str):
        if not isinstance(external_oid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", external_oid):
            raise ValueError("External order ID must be 1-32 ASCII letters, digits, underscores or hyphens")

    async def place_bracket_order(self, symbol: str, direction: str, notional_usdt, entry,
                                  stop_loss, take_profit, leverage: int, external_oid: str) -> dict:
        symbol = normalize_symbol(symbol)
        self._validate_external_id(external_oid)
        direction = str(direction).upper()
        if direction not in {"LONG", "SHORT"}:
            raise ValueError("Direction must be LONG or SHORT")
        contract = await self.get_contract(symbol, refresh=True)
        if contract.get("apiAllowed") is not True or contract.get("state") != 0:
            raise ValueError("Contract currently disallows API trading")
        if contract.get("positionOpenType") not in (1, 3):
            raise ValueError("Contract does not support isolated margin")
        if isinstance(leverage, bool) or int(leverage) != leverage or not int(contract["minLeverage"]) <= int(leverage) <= int(contract["maxLeverage"]):
            raise ValueError("Leverage outside contract limits")
        country_max = int(contract.get("countryConfigContractMaxLeverage") or 0)
        if country_max > 0 and leverage > country_max:
            raise ValueError("Leverage outside account country limits")
        long = direction == "LONG"
        entry_price, stop, take = round_bracket_prices(contract, direction, entry, stop_loss, take_profit)
        notional = _decimal(notional_usdt, "notional")
        # Outward stop rounding must not increase the source setup's cash risk.
        original_entry = _decimal(entry, "entry")
        original_risk = notional * abs(original_entry - _decimal(stop_loss, "stop loss")) / original_entry
        adjusted_notional = original_risk * entry_price / abs(entry_price - stop)
        volume = contracts_for_notional(contract, min(notional, adjusted_notional), entry_price)
        position_mode = await self.get_position_mode()
        payload = {"symbol": symbol, "price": entry_price, "vol": volume, "leverage": int(leverage),
                   "side": 1 if long else 3, "type": 1, "openType": 1, "externalOid": external_oid,
                   "positionMode": position_mode, "stopLossPrice": stop, "takeProfitPrice": take,
                   "lossTrend": 2 if contract.get("stopOnlyFair") is True else 1,
                   "profitTrend": 2 if contract.get("stopOnlyFair") is True else 1, "priceProtect": 0}
        result = await self._request("POST", ep.ORDER, payload)
        if not isinstance(result, dict) or not result.get("orderId"):
            raise MEXCAPIError(-2, uncertain=True)
        return {**result, "externalOid": external_oid, "symbol": symbol,
                "vol": str(volume), "contracts": str(volume), "contractSize": str(contract["contractSize"]),
                "price": str(entry_price), "stopLossPrice": str(stop), "takeProfitPrice": str(take),
                "positionMode": position_mode,
                "notionalUsdt": str(volume * _decimal(contract["contractSize"], "contract size") * entry_price),
                "riskUsdt": str(volume * _decimal(contract["contractSize"], "contract size") * abs(entry_price - stop)),
                "riskReward": str(abs(take - entry_price) / abs(entry_price - stop))}

    async def close_position(self, position: dict, external_oid: str) -> dict:
        self._validate_external_id(external_oid)
        symbol = normalize_symbol(position["symbol"])
        if int(position.get("state", 0)) != 1 or int(position["positionType"]) not in (1, 2):
            raise ValueError("Position is not a closable long or short")
        contract = await self.get_contract(symbol, refresh=True)
        held = _decimal(position["holdVol"], "held volume")
        frozen = _decimal(position.get("frozenVol", 0), "frozen volume", positive=False)
        if not 0 <= frozen <= held:
            raise ValueError("Invalid frozen position volume")
        volume = held - frozen
        volume = _round_step(volume, _decimal(contract["volUnit"], "volume step"))
        if volume < _decimal(contract["minVol"], "minimum volume"):
            raise ValueError("No available position volume to close")
        ticker = await self.get_ticker(symbol)
        mode = await self.get_position_mode()
        payload = {"symbol": symbol, "positionId": position["positionId"], "price": _decimal(ticker["lastPrice"], "last price"),
                   "vol": volume, "side": 4 if int(position["positionType"]) == 1 else 2,
                   "type": 5, "openType": int(position["openType"]), "positionMode": mode, "externalOid": external_oid}
        if mode == 2:
            payload["reduceOnly"] = True
        result = await self._request("POST", ep.ORDER, payload)
        if not isinstance(result, dict) or not result.get("orderId"):
            raise MEXCAPIError(-2, uncertain=True)
        return {**result, "vol": str(volume), "externalOid": external_oid}

    async def cancel_order(self, symbol: str, order_id: str):
        normalize_symbol(symbol)
        if not str(order_id).isdigit():
            raise ValueError("Invalid order ID")
        result = await self._request("POST", ep.CANCEL, [int(order_id)])
        if not isinstance(result, list) or len(result) != 1:
            raise MEXCAPIError(-2, uncertain=True)
        if int(result[0].get("errorCode", -1)) != 0:
            raise MEXCAPIError(int(result[0].get("errorCode", -1)))
        return result[0]

    async def set_leverage(self, symbol: str, side: str, leverage: int):
        direction = str(side).upper()
        if direction not in {"LONG", "SHORT"}:
            raise ValueError("Direction must be LONG or SHORT")
        return await self._request("POST", ep.LEVERAGE, {"symbol": normalize_symbol(symbol), "positionType": 1 if direction == "LONG" else 2,
                                                        "openType": 1, "leverage": int(leverage)})

    async def close(self):
        await self.client.aclose()
