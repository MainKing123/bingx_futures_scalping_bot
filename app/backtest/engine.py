from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

import pandas as pd

from app.backtest.profiles import StrategyProfile
from app.backtest.schemas import BacktestSymbolResult, BacktestTradeLogItem
from app.config import Settings
from app.exchange.client import BingXClient
from app.risk.risk_manager import RiskManager
from app.schemas.setup import TradeSetup
from app.strategy.crt_ict import analyze_crt_ict_from_df
from app.strategy.multi_tf import analyze_htf_from_df, find_ltf_entry_from_df

SignalProvider = Callable[
    [str, pd.DataFrame, pd.DataFrame, Settings, RiskManager, datetime],
    Awaitable[TradeSetup | None],
]
SignalAtIndexProvider = Callable[[int, datetime, RiskManager], Awaitable[TradeSetup | None]]

INTERVAL_TO_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
}


def _interval_minutes(interval: str) -> int:
    return INTERVAL_TO_MINUTES.get(interval, 5)


def _bars_for_days(days: int, interval: str, extra: int = 300) -> int:
    return max(200, int(days * 1440 / _interval_minutes(interval)) + extra)


class BacktestEngine:
    def __init__(self, client: BingXClient, settings: Settings, signal_provider: SignalProvider | None = None):
        self.client = client
        self.settings = settings
        self.signal_provider = signal_provider or self._default_signal_provider

    async def _default_signal_provider(
        self,
        symbol: str,
        ltf_df: pd.DataFrame,
        htf_df: pd.DataFrame,
        settings: Settings,
        risk_manager: RiskManager,
        now: datetime,
    ) -> TradeSetup | None:
        htf = analyze_htf_from_df(htf_df)
        return find_ltf_entry_from_df(
            symbol=symbol,
            htf=htf,
            ltf_df=ltf_df,
            settings=settings,
            risk_manager=risk_manager,
            now=now,
            enforce_session_filter=False,
        )

    @staticmethod
    def _apply_entry_slippage(price: float, direction: str, slippage_bps: float) -> float:
        slip = slippage_bps / 10_000
        if direction == "LONG":
            return price * (1 + slip)
        return price * (1 - slip)

    @staticmethod
    def _apply_exit_slippage(price: float, direction: str, slippage_bps: float) -> float:
        slip = slippage_bps / 10_000
        if direction == "LONG":
            return price * (1 - slip)
        return price * (1 + slip)

    @staticmethod
    def _resolve_exit(setup: TradeSetup, high: float, low: float) -> tuple[str, float] | None:
        if setup.direction == "LONG":
            tp_event: tuple[str, float] | None = None
            if high >= setup.take_profits[2]:
                tp_event = ("TP3_HIT", setup.take_profits[2])
            elif high >= setup.take_profits[1]:
                tp_event = ("TP2_HIT", setup.take_profits[1])
            elif high >= setup.take_profits[0]:
                tp_event = ("TP1_HIT", setup.take_profits[0])
            sl_hit = low <= setup.stop_loss
            if sl_hit and tp_event:
                return ("SL_HIT", setup.stop_loss)
            if sl_hit:
                return ("SL_HIT", setup.stop_loss)
            if tp_event:
                return tp_event
            return None

        tp_event = None
        if low <= setup.take_profits[2]:
            tp_event = ("TP3_HIT", setup.take_profits[2])
        elif low <= setup.take_profits[1]:
            tp_event = ("TP2_HIT", setup.take_profits[1])
        elif low <= setup.take_profits[0]:
            tp_event = ("TP1_HIT", setup.take_profits[0])
        sl_hit = high >= setup.stop_loss
        if sl_hit and tp_event:
            return ("SL_HIT", setup.stop_loss)
        if sl_hit:
            return ("SL_HIT", setup.stop_loss)
        if tp_event:
            return tp_event
        return None

    @staticmethod
    def _calc_pnl_percent(entry: float, exit_price: float, direction: str, fee_bps: float) -> float:
        gross_pct = ((exit_price - entry) / entry * 100) if direction == "LONG" else ((entry - exit_price) / entry * 100)
        fee_pct = (2 * fee_bps) / 100
        return gross_pct - fee_pct

    @staticmethod
    def _max_drawdown(pnl_series: list[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnl_series:
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        return max_dd

    @staticmethod
    def _empty_result(symbol: str) -> BacktestSymbolResult:
        return BacktestSymbolResult(
            symbol=symbol,
            trades_count=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            expectancy=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            total_pnl_percent=0.0,
        )

    @staticmethod
    def _build_runtime_settings(base: Settings, profile: StrategyProfile, ltf_timeframe: str, htf_timeframe: str) -> Settings:
        runtime = base.model_copy(deep=True)
        runtime.min_risk_reward = profile.min_risk_reward
        runtime.swing_lookback = profile.swing_lookback
        runtime.ob_max_age_candles = profile.ob_max_age_candles
        runtime.min_confluences = profile.min_confluences
        runtime.max_poi_distance_pct = profile.max_poi_distance_pct
        runtime.active_sessions = ["all"]
        runtime.htf_timeframe = htf_timeframe
        runtime.ltf_timeframe = ltf_timeframe
        return runtime

    async def _simulate(
        self,
        *,
        ltf_df: pd.DataFrame,
        runtime_settings: Settings,
        fee_bps: float,
        slippage_bps: float,
        cooldown_candles: int,
        warmup: int,
        setup_provider: SignalAtIndexProvider,
    ) -> BacktestSymbolResult:
        risk_manager = RiskManager(runtime_settings)
        cooldown_until_idx = -1
        active_trade: dict | None = None
        trade_log: list[BacktestTradeLogItem] = []
        pnl_series: list[float] = []

        for idx in range(warmup, len(ltf_df) - 1):
            row = ltf_df.iloc[idx]
            row_time = ltf_df.index[idx].to_pydatetime()
            high = float(row["high"])
            low = float(row["low"])

            if active_trade is not None:
                exit_event = self._resolve_exit(active_trade["setup"], high, low)
                if exit_event is not None:
                    status, exit_base = exit_event
                    exit_price = self._apply_exit_slippage(exit_base, active_trade["setup"].direction, slippage_bps)
                    pnl_pct = self._calc_pnl_percent(active_trade["entry_price"], exit_price, active_trade["setup"].direction, fee_bps)
                    pnl_series.append(pnl_pct)
                    risk_manager.record_result(runtime_settings.account_balance_usdt * pnl_pct / 100)
                    risk_manager.open_setups = max(0, risk_manager.open_setups - 1)
                    trade_log.append(
                        BacktestTradeLogItem(
                            signal_time=active_trade["signal_time"],
                            entry_time=active_trade["entry_time"],
                            exit_time=row_time,
                            direction=active_trade["setup"].direction,
                            status=status,
                            entry_price=round(active_trade["entry_price"], 6),
                            exit_price=round(exit_price, 6),
                            pnl_percent=round(pnl_pct, 6),
                        )
                    )
                    active_trade = None
                    cooldown_until_idx = idx + cooldown_candles

            if active_trade is not None:
                continue
            if idx <= cooldown_until_idx:
                continue
            if not risk_manager.can_open_setup():
                continue

            setup = await setup_provider(idx, row_time, risk_manager)
            if setup is None:
                continue

            next_open = float(ltf_df.iloc[idx + 1]["open"])
            entry_price = self._apply_entry_slippage(next_open, setup.direction, slippage_bps)
            active_trade = {
                "setup": setup,
                "signal_time": row_time,
                "entry_time": ltf_df.index[idx + 1].to_pydatetime(),
                "entry_price": entry_price,
            }
            risk_manager.open_setups += 1

        wins = sum(1 for pnl in pnl_series if pnl > 0)
        losses = sum(1 for pnl in pnl_series if pnl < 0)
        gross_profit = sum(max(pnl, 0.0) for pnl in pnl_series)
        gross_loss = sum(min(pnl, 0.0) for pnl in pnl_series)
        trades_count = len(pnl_series)
        win_rate = (wins / trades_count * 100) if trades_count else 0.0
        expectancy = sum(pnl_series) / trades_count if trades_count else 0.0
        if gross_loss < 0:
            profit_factor = gross_profit / abs(gross_loss)
        else:
            profit_factor = gross_profit if gross_profit > 0 else 0.0
        total_pnl = sum(pnl_series)
        max_drawdown = self._max_drawdown(pnl_series)

        return BacktestSymbolResult(
            symbol="",
            trades_count=trades_count,
            wins=wins,
            losses=losses,
            win_rate=round(win_rate, 4),
            expectancy=round(expectancy, 6),
            profit_factor=round(profit_factor, 6),
            max_drawdown=round(max_drawdown, 6),
            total_pnl_percent=round(total_pnl, 6),
            trade_log=trade_log,
        )

    async def run_symbol_legacy(
        self,
        symbol: str,
        *,
        lookback_days: int,
        ltf_timeframe: str,
        htf_timeframe: str,
        profile: StrategyProfile,
        fee_bps: float,
        slippage_bps: float,
        cooldown_candles: int,
    ) -> BacktestSymbolResult:
        ltf_limit = _bars_for_days(lookback_days, ltf_timeframe)
        htf_limit = _bars_for_days(lookback_days, htf_timeframe)
        ltf_df = await self.client.get_klines(symbol, interval=ltf_timeframe, limit=ltf_limit)
        htf_df = await self.client.get_klines(symbol, interval=htf_timeframe, limit=htf_limit)
        if ltf_df.empty or htf_df.empty or len(ltf_df) < 30:
            return self._empty_result(symbol)

        runtime_settings = self._build_runtime_settings(self.settings, profile, ltf_timeframe, htf_timeframe)
        warmup = max(25, runtime_settings.swing_lookback * 6)

        async def provider(idx: int, row_time: datetime, risk_manager: RiskManager) -> TradeSetup | None:
            ltf_slice = ltf_df.iloc[: idx + 1]
            htf_slice = htf_df[htf_df.index <= ltf_df.index[idx]]
            if len(htf_slice) < 20:
                return None
            return await self.signal_provider(symbol, ltf_slice, htf_slice, runtime_settings, risk_manager, row_time)

        result = await self._simulate(
            ltf_df=ltf_df,
            runtime_settings=runtime_settings,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            cooldown_candles=cooldown_candles,
            warmup=warmup,
            setup_provider=provider,
        )
        result.symbol = symbol
        return result

    async def run_symbol_crt_ict(
        self,
        symbol: str,
        *,
        lookback_days: int,
        ltf_timeframe: str,
        profile: StrategyProfile,
        fee_bps: float,
        slippage_bps: float,
        cooldown_candles: int,
    ) -> BacktestSymbolResult:
        ltf_limit = _bars_for_days(lookback_days, ltf_timeframe)
        htf_4h_limit = _bars_for_days(lookback_days, "4h", extra=120)
        htf_1d_limit = _bars_for_days(lookback_days, "1d", extra=60)
        ltf_df = await self.client.get_klines(symbol, interval=ltf_timeframe, limit=ltf_limit)
        htf_4h_df = await self.client.get_klines(symbol, interval="4h", limit=htf_4h_limit)
        htf_1d_df = await self.client.get_klines(symbol, interval="1d", limit=htf_1d_limit)
        if ltf_df.empty or htf_4h_df.empty or htf_1d_df.empty or len(ltf_df) < 40:
            return self._empty_result(symbol)

        runtime_settings = self._build_runtime_settings(self.settings, profile, ltf_timeframe, "4h")
        runtime_settings.crt_entry_timeframes = [ltf_timeframe]
        warmup = max(30, runtime_settings.swing_lookback * 6, runtime_settings.crt_range_lookback + 2)

        async def provider(idx: int, row_time: datetime, _: RiskManager) -> TradeSetup | None:
            ltf_slice = ltf_df.iloc[: idx + 1]
            htf_4h_slice = htf_4h_df[htf_4h_df.index <= ltf_df.index[idx]]
            htf_1d_slice = htf_1d_df[htf_1d_df.index <= ltf_df.index[idx]]
            if len(htf_4h_slice) < 20 or len(htf_1d_slice) < 20:
                return None
            _, setup = analyze_crt_ict_from_df(symbol, ltf_slice, htf_4h_slice, htf_1d_slice, runtime_settings, row_time)
            return setup

        result = await self._simulate(
            ltf_df=ltf_df,
            runtime_settings=runtime_settings,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            cooldown_candles=cooldown_candles,
            warmup=warmup,
            setup_provider=provider,
        )
        result.symbol = symbol
        return result

    async def run_symbol(
        self,
        symbol: str,
        *,
        lookback_days: int,
        ltf_timeframe: str,
        htf_timeframe: str,
        profile: StrategyProfile,
        fee_bps: float,
        slippage_bps: float,
        cooldown_candles: int,
        strategy: str = "legacy_choch_ob",
    ) -> BacktestSymbolResult:
        if strategy == "crt_ict":
            return await self.run_symbol_crt_ict(
                symbol,
                lookback_days=lookback_days,
                ltf_timeframe=ltf_timeframe,
                profile=profile,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                cooldown_candles=cooldown_candles,
            )
        return await self.run_symbol_legacy(
            symbol,
            lookback_days=lookback_days,
            ltf_timeframe=ltf_timeframe,
            htf_timeframe=htf_timeframe,
            profile=profile,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            cooldown_candles=cooldown_candles,
        )
