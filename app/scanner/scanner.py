from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from loguru import logger

from app.config import Settings
from app.exchange.client import BingXClient
from app.exchange.ws_stream import BingXKlineStream
from app.schemas.setup import HTFAnalysis
from app.strategy.multi_tf import analyze_htf
from app.strategy.smc_engine import SMCEngine


@dataclass
class WatchlistEntry:
    symbol: str
    htf_analysis: HTFAnalysis
    poi_zone: tuple[float, float]
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_checked: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PairScanner:
    def __init__(self, client: BingXClient, kline_stream: BingXKlineStream, engine: SMCEngine, settings: Settings):
        self.client = client
        self.kline_stream = kline_stream
        self.engine = engine
        self.settings = settings
        self.watchlist: dict[str, WatchlistEntry] = {}
        self.sem = asyncio.Semaphore(5)

    async def scan(self):
        symbols = await self.client.get_top_symbols(self.settings.top_pairs_count, self.settings.min_daily_volume_usd)
        async def analyze(symbol: str):
            async with self.sem:
                try:
                    htf = await analyze_htf(self.client, symbol)
                    if htf.poi_zones:
                        zone = htf.poi_zones[0]
                        self.watchlist[symbol] = WatchlistEntry(symbol=symbol, htf_analysis=htf, poi_zone=(zone.zone_low, zone.zone_high))
                except Exception as e:
                    logger.exception(f"scan failed for {symbol}: {e}")
        await asyncio.gather(*(analyze(s) for s in symbols))

    async def watch_loop(self):
        while True:
            now = datetime.now(timezone.utc)
            expired = [s for s, v in self.watchlist.items() if now - v.added_at > timedelta(hours=self.settings.watchlist_max_age_hours)]
            for symbol in expired:
                self.watchlist.pop(symbol, None)
            await asyncio.sleep(60)
