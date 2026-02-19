from __future__ import annotations

from app.config import Settings
from app.exchange.client import BingXClient
from app.risk.risk_manager import RiskManager
from app.schemas.analysis import CRTICTAnalysisResponse
from app.schemas.setup import MarketOverview, SymbolAnalysis, TradeSetup
from app.strategy.crt_ict import analyze_crt_ict_from_df
from app.strategy.fvg import find_fvg
from app.strategy.market_structure import detect_bos, detect_choch, determine_trend, find_swing_points
from app.strategy.multi_tf import analyze_htf, find_ltf_entry
from app.strategy.order_blocks import find_order_blocks


class SMCEngine:
    def __init__(self, client: BingXClient, settings: Settings, risk_manager: RiskManager):
        self.client = client
        self.settings = settings
        self.risk_manager = risk_manager

    async def analyze_symbol(self, symbol: str) -> SymbolAnalysis:
        df = await self.client.get_klines(symbol, self.settings.htf_timeframe, 200)
        swings = find_swing_points(df, lookback=self.settings.swing_lookback)
        trend = determine_trend(swings)
        structure = detect_bos(df, swings, trend) + detect_choch(df, swings, trend)
        obs = find_order_blocks(df, structure, self.settings.ob_max_age_candles)
        fvgs = find_fvg(df, self.settings.fvg_min_size_percent)
        return SymbolAnalysis(symbol=symbol, trend=trend, swings=swings, structure=structure, order_blocks=obs, fvgs=fvgs)

    async def check_for_setup(self, symbol: str) -> TradeSetup | None:
        return await self.check_for_setup_legacy(symbol)

    async def check_for_setup_legacy(self, symbol: str) -> TradeSetup | None:
        htf = await analyze_htf(self.client, symbol)
        return await find_ltf_entry(self.client, symbol, htf, self.settings, self.risk_manager)

    async def analyze_crt_ict(self, symbol: str, ltf_timeframe: str | None = None) -> tuple[CRTICTAnalysisResponse, TradeSetup | None]:
        allowed_ltf = list(self.settings.crt_entry_timeframes or ["5m", "15m"])
        timeframe = (ltf_timeframe or allowed_ltf[0]).lower()
        if timeframe not in allowed_ltf:
            timeframe = allowed_ltf[0]
        ltf_df = await self.client.get_klines(symbol, timeframe, 600)
        htf_4h_df = await self.client.get_klines(symbol, "4h", 500)
        htf_1d_df = await self.client.get_klines(symbol, "1d", 200)
        runtime_settings = self.settings.model_copy(deep=True)
        runtime_settings.crt_entry_timeframes = [timeframe]
        return analyze_crt_ict_from_df(symbol, ltf_df, htf_4h_df, htf_1d_df, runtime_settings)

    async def get_market_overview(self, symbol: str) -> MarketOverview:
        analysis = await self.analyze_symbol(symbol)
        return MarketOverview(symbol=symbol, trend=analysis.trend.direction, bias=analysis.trend.direction, active_obs=analysis.order_blocks, active_fvgs=analysis.fvgs)
