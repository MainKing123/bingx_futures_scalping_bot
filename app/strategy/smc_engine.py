from __future__ import annotations

from app.config import Settings
from app.exchange.client import BingXClient
from app.schemas.setup import MarketOverview, SymbolAnalysis, TradeSetup
from app.strategy.fvg import find_fvg
from app.strategy.market_structure import detect_bos, detect_choch, determine_trend, find_swing_points
from app.strategy.multi_tf import analyze_htf, find_ltf_entry
from app.strategy.order_blocks import find_order_blocks


class SMCEngine:
    def __init__(self, client: BingXClient, settings: Settings):
        self.client = client
        self.settings = settings

    async def analyze_symbol(self, symbol: str) -> SymbolAnalysis:
        df = await self.client.get_klines(symbol, self.settings.htf_timeframe, 200)
        swings = find_swing_points(df, lookback=self.settings.swing_lookback)
        trend = determine_trend(swings)
        structure = detect_bos(df, swings, trend) + detect_choch(df, swings, trend)
        obs = find_order_blocks(df, structure, self.settings.ob_max_age_candles)
        fvgs = find_fvg(df, self.settings.fvg_min_size_percent)
        return SymbolAnalysis(symbol=symbol, trend=trend, swings=swings, structure=structure, order_blocks=obs, fvgs=fvgs)

    async def check_for_setup(self, symbol: str) -> TradeSetup | None:
        htf = await analyze_htf(self.client, symbol)
        return await find_ltf_entry(self.client, symbol, htf)

    async def get_market_overview(self, symbol: str) -> MarketOverview:
        analysis = await self.analyze_symbol(symbol)
        return MarketOverview(symbol=symbol, trend=analysis.trend.direction, bias=analysis.trend.direction, active_obs=analysis.order_blocks, active_fvgs=analysis.fvgs)
