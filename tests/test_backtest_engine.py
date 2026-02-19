import asyncio
from datetime import datetime, timezone

import pandas as pd

from app.backtest.engine import BacktestEngine
from app.backtest.profiles import StrategyProfile
from app.config import Settings
from app.schemas.setup import TradeSetup


def _make_df(pattern: str = "tp") -> pd.DataFrame:
    idx = pd.date_range(datetime(2025, 1, 1, tzinfo=timezone.utc), periods=40, freq="min")
    rows = []
    for _ in range(40):
        rows.append((100.0, 101.0, 99.5, 100.0))

    rows[31] = (101.0, 102.0, 100.0, 101.0)
    if pattern == "tp":
        rows[32] = (101.0, 110.5, 100.2, 109.5)
    elif pattern == "collision":
        rows[32] = (101.0, 110.5, 97.5, 99.0)

    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


class FakeClient:
    def __init__(self, ltf_df: pd.DataFrame, htf_df: pd.DataFrame | None = None):
        self.ltf_df = ltf_df
        self.htf_df = htf_df if htf_df is not None else ltf_df

    async def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time: int | None = None, end_time: int | None = None):
        return self.htf_df.copy() if interval in {"30m", "4h", "1d"} else self.ltf_df.copy()


def _setup_provider(stop_loss: float = 98.0):
    async def provider(symbol, ltf_df, htf_df, settings, risk_manager, now):
        if len(ltf_df) == 31:
            return TradeSetup(
                timestamp=now,
                symbol=symbol,
                direction="LONG",
                setup_type="CHOCH_OB",
                htf_bias="BULLISH",
                entry=100.0,
                stop_loss=stop_loss,
                take_profits=[110.0, 112.0, 114.0],
                risk_reward=2.0,
                confidence="HIGH",
                confluences=["test"],
                position_size_usdt=100.0,
            )
        return None

    return provider


def _profile() -> StrategyProfile:
    return StrategyProfile(
        name="default",
        min_risk_reward=2.0,
        swing_lookback=3,
        ob_max_age_candles=50,
        min_confluences=1,
        max_poi_distance_pct=1.0,
    )


def test_no_lookahead_entry_is_next_bar_open():
    async def run():
        engine = BacktestEngine(FakeClient(_make_df("tp")), Settings(), signal_provider=_setup_provider())
        result = await engine.run_symbol(
            "BTC-USDT",
            lookback_days=1,
            ltf_timeframe="1m",
            htf_timeframe="30m",
            profile=_profile(),
            fee_bps=0.0,
            slippage_bps=0.0,
            cooldown_candles=0,
        )
        assert result.trades_count == 1
        assert result.trade_log[0].entry_time == _make_df("tp").index[31].to_pydatetime()
        assert result.trade_log[0].status == "TP1_HIT"
        assert result.trade_log[0].pnl_percent > 0

    asyncio.run(run())


def test_sl_tp_collision_prioritizes_stop_loss():
    async def run():
        engine = BacktestEngine(FakeClient(_make_df("collision")), Settings(), signal_provider=_setup_provider(stop_loss=98.0))
        result = await engine.run_symbol(
            "BTC-USDT",
            lookback_days=1,
            ltf_timeframe="1m",
            htf_timeframe="30m",
            profile=_profile(),
            fee_bps=0.0,
            slippage_bps=0.0,
            cooldown_candles=0,
        )
        assert result.trades_count == 1
        assert result.trade_log[0].status == "SL_HIT"
        assert result.trade_log[0].pnl_percent < 0

    asyncio.run(run())


def test_fee_and_slippage_reduce_expectancy():
    async def run():
        settings = Settings()
        low_cost_engine = BacktestEngine(FakeClient(_make_df("tp")), settings, signal_provider=_setup_provider())
        high_cost_engine = BacktestEngine(FakeClient(_make_df("tp")), settings, signal_provider=_setup_provider())

        low_cost = await low_cost_engine.run_symbol(
            "BTC-USDT",
            lookback_days=1,
            ltf_timeframe="1m",
            htf_timeframe="30m",
            profile=_profile(),
            fee_bps=0.0,
            slippage_bps=0.0,
            cooldown_candles=0,
        )
        high_cost = await high_cost_engine.run_symbol(
            "BTC-USDT",
            lookback_days=1,
            ltf_timeframe="1m",
            htf_timeframe="30m",
            profile=_profile(),
            fee_bps=12.0,
            slippage_bps=12.0,
            cooldown_candles=0,
        )
        assert high_cost.expectancy < low_cost.expectancy

    asyncio.run(run())


def test_crt_ict_no_lookahead_entry_is_next_bar_open(monkeypatch):
    async def run():
        async def fake_signal(*args, **kwargs):
            return None

        def fake_analyze(symbol, ltf_df, htf_4h_df, htf_1d_df, settings, now):
            if len(ltf_df) == 31:
                setup = TradeSetup(
                    timestamp=now,
                    symbol=symbol,
                    direction="LONG",
                    setup_type="CRT_ICT",
                    htf_bias="BULLISH",
                    entry=100.0,
                    stop_loss=98.0,
                    take_profits=[110.0, 112.0, 114.0],
                    risk_reward=3.0,
                    confidence="HIGH",
                    confluences=["CRT", "MSS"],
                    position_size_usdt=100.0,
                )
                return None, setup
            return None, None

        monkeypatch.setattr("app.backtest.engine.analyze_crt_ict_from_df", fake_analyze)

        engine = BacktestEngine(FakeClient(_make_df("tp")), Settings(), signal_provider=fake_signal)
        result = await engine.run_symbol(
            "BTC-USDT",
            lookback_days=1,
            ltf_timeframe="5m",
            htf_timeframe="4h",
            profile=_profile(),
            fee_bps=0.0,
            slippage_bps=0.0,
            cooldown_candles=0,
            strategy="crt_ict",
        )
        assert result.trades_count == 1
        assert result.trade_log[0].entry_time == _make_df("tp").index[31].to_pydatetime()
        assert result.trade_log[0].status == "TP1_HIT"

    asyncio.run(run())
