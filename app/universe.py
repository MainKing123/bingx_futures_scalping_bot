from __future__ import annotations

import asyncio
import copy
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from app.config import Settings
from app.exchange.client import MEXCClient, normalize_symbol

MARKET_CAP_URL = "https://api.coingecko.com/api/v3/coins/markets"
CAP_MAX_AGE_SECONDS = 6 * 3600
LOOKBACK = 96
INTERVAL = "15m"
INTERVAL_SECONDS = 900
STABLE_IDS = {
    "tether", "usd-coin", "dai", "usds", "ethena-usde", "ethena-staked-usde",
    "first-digital-usd", "true-usd", "binance-usd", "paypal-usd", "pax-dollar",
    "usd1-wlfi", "world-liberty-financial-usd", "frax", "frax-ether", "usual-usd",
    "usdd", "pax-gold", "tether-gold", "figure-heloc",
}
STABLE_SYMBOLS = {"USDT", "USDC", "DAI", "USDS", "SUSDS", "USDE", "SUSDE", "USD1",
                  "FDUSD", "TUSD", "BUSD", "PYUSD", "USDP", "USDD", "FRAX", "USDF", "USD0", "PAXG", "XAUT"}
DERIVATIVE_IDS = {"staked-ether", "wrapped-steth", "coinbase-wrapped-btc", "wrapped-bitcoin",
                  "binance-peg-ethereum", "binance-staked-sol", "binance-wrapped-btc"}


class UniverseSelectionError(Exception):
    """A complete, verified set of five is unavailable; scanning must pause."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _excluded_coin(coin: dict) -> bool:
    identifier = str(coin.get("id", "")).lower()
    symbol = str(coin.get("symbol", "")).upper()
    name = str(coin.get("name", "")).lower()
    return (identifier in STABLE_IDS or symbol in STABLE_SYMBOLS or identifier in DERIVATIVE_IDS
            or any(tag in identifier or tag in name for tag in ("wrapped", "bridged", "staked", "restaked", "stablecoin")))


class UniverseSelector:
    def __init__(self, client: MEXCClient, settings: Settings, *,
                 cap_transport: httpx.AsyncBaseTransport | None = None,
                 cache_path: str | Path = ".cache/universe.json"):
        self.client = client
        self.settings = settings
        self.cap_client = httpx.AsyncClient(timeout=20.0, transport=cap_transport)
        self.cache_path = Path(cache_path)
        self.lock = asyncio.Lock()
        self.snapshot: dict | None = self._load_cache()

    def _policy(self) -> dict:
        return {"pair_selection": self.settings.pair_selection, "pair_count": self.settings.pair_count,
                "large_cap_top_n": self.settings.large_cap_top_n,
                "min_pair_volume_usd": self.settings.min_pair_volume_usd,
                "lookback": LOOKBACK, "interval": INTERVAL,
                "fixed_symbols": list(self.settings.trading_symbols) if self.settings.pair_selection == "fixed" else []}

    @staticmethod
    def _age(timestamp: str | None, now: datetime) -> float:
        try:
            date = datetime.fromisoformat(timestamp)
            if date.tzinfo is None:
                return math.inf
            age = (now - date).total_seconds()
            return age if age >= -60 else math.inf
        except (ValueError, TypeError):
            return math.inf

    def _load_cache(self) -> dict | None:
        try:
            if self.cache_path.stat().st_size > 512_000:
                return None
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) and data.get("version") == 1 else None
        except (OSError, ValueError):
            return None

    def _save(self, snapshot: dict):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.cache_path)
        self.snapshot = snapshot

    @staticmethod
    def _valid_rows(rows, *, dynamic: bool, top_n: int, minimum_volume: float) -> bool:
        if not isinstance(rows, list) or len(rows) != 5:
            return False
        try:
            if len({normalize_symbol(row["symbol"]) for row in rows}) != 5:
                return False
            for row in rows:
                if not row["symbol"].endswith("_USDT"):
                    return False
                if not all(math.isfinite(float(row[field])) and float(row[field]) > 0 for field in ("quoteVolume", "natr_percent")):
                    return False
                if float(row["quoteVolume"]) < minimum_volume:
                    return False
                if dynamic:
                    cap = float(row["market_cap_usd"])
                    if not 1 <= int(row["market_cap_rank"]) <= top_n or not math.isfinite(cap) or cap <= 0:
                        return False
            return True
        except (KeyError, ValueError, TypeError):
            return False

    def _fresh_selection(self, now: datetime) -> bool:
        if not self.snapshot or self.snapshot.get("parameters") != self._policy():
            return False
        return (self._age(self.snapshot.get("selected_at"), now) <= self.settings.universe_refresh_seconds
                and (self.settings.pair_selection == "fixed" or self._age(self.snapshot.get("market_cap_fetched_at"), now) <= CAP_MAX_AGE_SECONDS)
                and self._valid_rows(self.snapshot.get("selected"), dynamic=self.settings.pair_selection == "dynamic",
                                    top_n=self.settings.large_cap_top_n, minimum_volume=self.settings.min_pair_volume_usd))

    def _validated_caps(self, rows, *, now: datetime) -> list[dict]:
        if not isinstance(rows, list) or len(rows) < self.settings.large_cap_top_n:
            raise UniverseSelectionError("Market-cap response is incomplete")
        result, ranks = [], set()
        for coin in rows:
            if not isinstance(coin, dict):
                raise UniverseSelectionError("Market-cap response is malformed")
            try:
                rank = int(coin["market_cap_rank"])
                cap = float(coin["market_cap"])
                identifier = str(coin["id"])
                symbol = str(coin["symbol"]).upper()
            except (KeyError, TypeError, ValueError):
                continue
            if not 1 <= rank <= self.settings.large_cap_top_n:
                continue
            if not math.isfinite(cap) or cap <= 0 or not identifier:
                raise UniverseSelectionError("Large-cap record is invalid")
            if self._age(coin.get("last_updated"), now) > CAP_MAX_AGE_SECONDS:
                raise UniverseSelectionError("Large-cap source records are stale")
            ranks.add(rank)
            result.append({"id": identifier, "symbol": symbol, "name": str(coin.get("name", identifier)),
                           "market_cap_rank": rank, "market_cap": cap,
                           "last_updated": coin.get("last_updated")})
        if ranks != set(range(1, self.settings.large_cap_top_n + 1)):
            raise UniverseSelectionError("Market-cap ranking has missing top-ranked coins")
        return result

    async def _market_caps(self, now: datetime) -> tuple[list[dict], str, str]:
        try:
            response = await self.cap_client.get(MARKET_CAP_URL, params={"vs_currency": "usd", "order": "market_cap_desc",
                    "per_page": 100, "page": 1, "sparkline": "false"})
            if response.status_code != 200:
                raise UniverseSelectionError(f"Market-cap API unavailable (HTTP {response.status_code})")
            rows = self._validated_caps(response.json(), now=now)
            return rows, now.isoformat(), "live"
        except (httpx.HTTPError, ValueError, UniverseSelectionError):
            # Only a previously verified source snapshot is reused, never static coin guesses.
            if self.snapshot and self._age(self.snapshot.get("market_cap_fetched_at"), now) <= CAP_MAX_AGE_SECONDS:
                try:
                    rows = self._validated_caps(self.snapshot.get("market_cap_snapshot"), now=now)
                    return rows, self.snapshot["market_cap_fetched_at"], "cache"
                except UniverseSelectionError:
                    pass
            raise UniverseSelectionError("Fresh market-cap data unavailable; selection paused") from None

    @staticmethod
    def natr_percent(frame: pd.DataFrame, *, server_ms: int) -> float:
        if len(frame) < LOOKBACK + 1 or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise UniverseSelectionError("Insufficient closed candles for volatility ranking")
        frame = frame.tail(LOOKBACK + 1)
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise UniverseSelectionError("Candle timestamps are not ordered and unique")
        if not (frame.index.to_series().diff().dropna() == pd.Timedelta(seconds=INTERVAL_SECONDS)).all():
            raise UniverseSelectionError("Volatility candles contain gaps")
        expected_open_ms = ((server_ms - 1500) // (INTERVAL_SECONDS * 1000) - 1) * INTERVAL_SECONDS * 1000
        if int(frame.index[-1].timestamp() * 1000) != expected_open_ms:
            raise UniverseSelectionError("Volatility candles are stale or unclosed")
        try:
            close = pd.to_numeric(frame["close"], errors="raise")
            high = pd.to_numeric(frame["high"], errors="raise")
            low = pd.to_numeric(frame["low"], errors="raise")
        except (KeyError, TypeError, ValueError):
            raise UniverseSelectionError("Invalid volatility prices") from None
        if any(not values.map(math.isfinite).all() or values.le(0).any() for values in (close, high, low)):
            raise UniverseSelectionError("Invalid volatility prices")
        if high.lt(low).any() or high.lt(close).any() or low.gt(close).any():
            raise UniverseSelectionError("Inconsistent volatility OHLC")
        previous = close.shift(1)
        true_range = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
        score = float((true_range.iloc[-LOOKBACK:] / close.iloc[-LOOKBACK:]).mean() * 100)
        if not math.isfinite(score) or score <= 0:
            raise UniverseSelectionError("Volatility score unavailable")
        return score

    async def select(self, *, force_refresh: bool = False) -> list[dict]:
        async with self.lock:
            now = _utc_now()
            if not force_refresh and self._fresh_selection(now):
                return copy.deepcopy(self.snapshot["selected"])
            if self.settings.pair_count != 5:
                raise UniverseSelectionError("The universe must contain exactly five pairs")
            mode = self.settings.pair_selection
            cap_rows, cap_timestamp, cap_source = [], None, "not_used_fixed"
            if mode == "dynamic":
                cap_rows, cap_timestamp, cap_source = await self._market_caps(now)
            elif mode != "fixed":
                raise UniverseSelectionError("Unknown selection mode")
            contracts = await self.client.get_contracts()
            tickers = await self.client.get_tickers()
            allowed = {normalize_symbol(c["symbol"]): c for c in contracts
                       if c.get("apiAllowed") is True and c.get("state") == 0
                       and c.get("quoteCoin") == "USDT" and c.get("settleCoin") == "USDT"
                       and c.get("futureType", 1) == 1}
            quotes = {normalize_symbol(t["symbol"]): t for t in tickers}
            candidates = []
            if mode == "fixed":
                fixed = list(dict.fromkeys(normalize_symbol(s) for s in self.settings.trading_symbols))
                if len(fixed) != 5:
                    raise UniverseSelectionError("Fixed mode also requires exactly five unique pairs")
                rows = [{"symbol": s.removesuffix("_USDT"), "market_cap_rank": None, "market_cap": None} for s in fixed]
            else:
                counts = Counter(coin["symbol"] for coin in cap_rows)
                rows = [coin for coin in cap_rows if not _excluded_coin(coin) and counts[coin["symbol"]] == 1]
            for coin in rows:
                base = coin["symbol"]
                if not re.fullmatch(r"[A-Z0-9]+", base):
                    continue
                symbol = f"{base}_USDT"
                contract, ticker = allowed.get(symbol), quotes.get(symbol)
                if contract is None or ticker is None or str(contract.get("baseCoin", base)).upper() != base:
                    continue
                try:
                    volume = float(ticker["quoteVolume"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(volume) or volume < self.settings.min_pair_volume_usd:
                    continue
                candidates.append({"symbol": symbol, "market_cap_rank": coin["market_cap_rank"],
                                   "market_cap_usd": coin["market_cap"], "quoteVolume": volume,
                                   "coin_id": coin.get("id"), "coin_name": coin.get("name"),
                                   "market_cap_last_updated": coin.get("last_updated")})
            if len(candidates) < 5:
                raise UniverseSelectionError("Fewer than five eligible liquid large-cap MEXC futures")
            server_ms = await self.client.get_server_time()
            cutoff_ms = server_ms - 1500
            semaphore = asyncio.Semaphore(3)
            async def score(candidate):
                async with semaphore:
                    frame = await self.client.get_klines(candidate["symbol"], INTERVAL, limit=LOOKBACK + 1, end_time=cutoff_ms)
                    return {**candidate, "natr_percent": self.natr_percent(frame, server_ms=server_ms),
                            "candle_end": frame.index[-1].isoformat()}
            # A missing candidate prevents a valid top-five ranking; do not silently drop it.
            try:
                ranked = await asyncio.gather(*(score(c) for c in candidates))
            except Exception:
                raise UniverseSelectionError("Complete fresh volatility ranking unavailable; selection paused") from None
            ranked.sort(key=lambda row: (-row["natr_percent"], row["market_cap_rank"] or 0, row["symbol"]))
            selected = ranked[:5] if mode == "dynamic" else sorted(ranked, key=lambda row: fixed.index(row["symbol"]))
            snapshot = {"version": 1, "source": MARKET_CAP_URL if mode == "dynamic" else "explicit_fixed_configuration",
                        "selected_at": now.isoformat(), "market_cap_fetched_at": cap_timestamp,
                        "market_cap_source": cap_source, "market_cap_snapshot": cap_rows,
                        "market_data_server_time": pd.Timestamp(server_ms, unit="ms", tz="UTC").isoformat(),
                        "parameters": self._policy(), "candidates": ranked, "selected": selected}
            self._save(snapshot)
            return copy.deepcopy(selected)

    async def close(self):
        await self.cap_client.aclose()
