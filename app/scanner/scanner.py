from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from loguru import logger

from app.api.ws_manager import WSManager
from app.config import Settings
from app.db.engine import SessionLocal
from app.db.repository import increment_daily_stats, save_setup
from app.exchange.client import BingXClient
from app.exchange.ws_stream import BingXKlineStream
from app.notifications.telegram import TelegramNotifier
from app.risk.risk_manager import RiskManager
from app.schemas.setup import HTFAnalysis
from app.strategy.multi_tf import analyze_htf
from app.execution.executor import AutoExecutor
from app.strategy.smc_engine import SMCEngine
from app.tracking.trade_tracker import TradeTracker


@dataclass
class WatchlistEntry:
    symbol: str
    htf_analysis: HTFAnalysis
    poi_zone: tuple[float, float]
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_checked: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PairScanner:
    def __init__(
        self,
        client: BingXClient,
        kline_stream: BingXKlineStream,
        engine: SMCEngine,
        settings: Settings,
        ws_manager: WSManager,
        risk_manager: RiskManager,
        notifier: TelegramNotifier,
        trade_tracker: TradeTracker,
        executor: AutoExecutor | None = None,
    ):
        self.client = client
        self.kline_stream = kline_stream
        self.engine = engine
        self.settings = settings
        self.ws_manager = ws_manager
        self.risk_manager = risk_manager
        self.notifier = notifier
        self.trade_tracker = trade_tracker
        self.executor = executor
        self.watchlist: dict[str, WatchlistEntry] = {}
        self.sem = asyncio.Semaphore(5)
        self.kline_stream.on_kline_close(self._on_kline_close)

    async def _add_to_watchlist(self, symbol: str, htf: HTFAnalysis, zone: tuple[float, float]):
        if symbol in self.watchlist:
            self.watchlist[symbol].last_checked = datetime.now(timezone.utc)
            return
        self.watchlist[symbol] = WatchlistEntry(symbol=symbol, htf_analysis=htf, poi_zone=zone)
        await self.kline_stream.subscribe(symbol, self.settings.ltf_timeframe)
        await self.ws_manager.broadcast("watchlist_update", {"symbol": symbol, "action": "added", "poi_zone": zone})

    async def _remove_from_watchlist(self, symbol: str, reason: str):
        if symbol not in self.watchlist:
            return
        self.watchlist.pop(symbol, None)
        await self.kline_stream.unsubscribe(symbol, self.settings.ltf_timeframe)
        await self.ws_manager.broadcast("watchlist_update", {"symbol": symbol, "action": "removed", "reason": reason})

    async def _on_kline_close(self, symbol: str, interval: str, candle: dict):
        if interval != self.settings.ltf_timeframe or symbol not in self.watchlist:
            return
        try:
            setup = await self.engine.check_for_setup(symbol)
            if setup is None:
                return
            async with SessionLocal() as session:
                await save_setup(session, setup)
                await increment_daily_stats(session, setup)
            await self.ws_manager.broadcast("new_setup", setup.model_dump())
            await self.notifier.send_setup(setup)
            await self.trade_tracker.add(setup)
            if self.executor is not None and self.settings.auto_execution:
                await self.executor.execute_setup(setup)
            self.risk_manager.open_setups += 1
            logger.info(f"New setup detected for {symbol}: {setup.id}")
        except Exception as exc:
            logger.exception(f"Failed processing kline close for {symbol}: {exc}")

    async def scan(self):
        symbols = await self.client.get_top_symbols(self.settings.top_pairs_count, self.settings.min_daily_volume_usd)

        async def analyze(symbol: str):
            async with self.sem:
                try:
                    htf = await analyze_htf(self.client, symbol)
                    if not htf.poi_zones:
                        if symbol in self.watchlist:
                            await self._remove_from_watchlist(symbol, "poi_invalid")
                        return

                    zone = htf.poi_zones[0]
                    zone_low, zone_high = zone.zone_low, zone.zone_high
                    ticker = await self.client.get_ticker(symbol)
                    price = float(ticker.get("lastPrice") or 0)
                    zone_mid = (zone_low + zone_high) / 2
                    distance_pct = abs(price - zone_mid) / zone_mid * 100 if zone_mid else 999
                    if distance_pct < 0.5:
                        await self._add_to_watchlist(symbol, htf, (zone_low, zone_high))
                    elif symbol in self.watchlist:
                        await self._remove_from_watchlist(symbol, "price_far_from_poi")
                except Exception as e:
                    logger.exception(f"scan failed for {symbol}: {e}")

        await asyncio.gather(*(analyze(s) for s in symbols))

    async def watch_loop(self):
        while True:
            now = datetime.now(timezone.utc)
            for symbol, entry in list(self.watchlist.items()):
                if now - entry.added_at > timedelta(hours=self.settings.watchlist_max_age_hours):
                    await self._remove_from_watchlist(symbol, "expired")
                    continue
                ticker = await self.client.get_ticker(symbol)
                price = float(ticker.get("lastPrice") or 0)
                zone_mid = (entry.poi_zone[0] + entry.poi_zone[1]) / 2
                if zone_mid and abs(price - zone_mid) / zone_mid * 100 > 1:
                    await self._remove_from_watchlist(symbol, "price_far")
            await asyncio.sleep(60)
