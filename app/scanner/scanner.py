from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone

import pandas as pd
from loguru import logger
from sqlalchemy import select

from app.config import Settings
from app.db.engine import SessionLocal
from app.db.models import SetupRecord
from app.db.repository import save_setup, update_setup_status
from app.exchange.client import MEXCClient
from app.risk.risk_manager import RiskManager
from app.schemas.setup import TradeSetup
from app.strategy.volium import analyze_volium_from_df
from app.execution.economics import prepare_runtime_setup, current_contract_with_basis


class SignalScanner:
    """Poll closed MEXC candles and publish only the strategy from the video."""

    def __init__(self, client: MEXCClient, settings: Settings, risk_manager: RiskManager, tracker, executor, ws_manager=None, universe=None):
        self.client = client
        self.settings = settings
        self.risk_manager = risk_manager
        self.tracker = tracker
        self.executor = executor
        self.ws_manager = ws_manager
        self.universe = universe
        if settings.pair_selection == "dynamic" and universe is None:
            raise ValueError("Dynamic pair selection requires a market-cap universe selector")
        self.active_symbols = settings.trading_symbols if settings.pair_selection == "fixed" else []
        self.selected_pairs = []
        self.selection_error = None
        self.last_processed: dict[tuple[str, str], datetime] = {}
        self.last_scan_at: datetime | None = None
        self.last_errors: dict[str, str] = {}
        self.last_rejections: dict[str, str] = {}
        self.latest_signals: dict[str, TradeSetup | None] = {}
        self._publish_lock = asyncio.Lock()
        self._scan_lock = asyncio.Lock()
        self._fetch_semaphore = asyncio.Semaphore(5)

    def _timeframes(self, mode: str) -> tuple[str, ...]:
        if mode == "intraday":
            return ("1d", "1h", "5m")
        if mode == "scalp":
            return ("1h", "5m", "1m")
        if mode == "swing":
            return ("1w", "4h") if self.settings.volium_swing_context == "1w" else ("1d", "1h")
        raise ValueError(f"Unsupported strategy mode: {mode}")

    @staticmethod
    def _latest_timestamp(frame: pd.DataFrame) -> datetime:
        timestamp = pd.Timestamp(frame.index[-1])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC").to_pydatetime()

    async def _publish_setup(self, setup: TradeSetup) -> bool:
        # The strategy fingerprints the mode, symbol, and swept context candle.
        # Keep that ID across scans and restarts, including after a closed trade.
        async with self._publish_lock:
            if not self.risk_manager.can_open_setup():
                logger.info(f"Risk guard blocked setup for {setup.symbol}")
                return False
            if self.settings.auto_execution and self.executor is None:
                logger.error("Automatic execution requires an executor")
                return False
            guarded = getattr(self.settings, "execution_cost_guard_enabled", False)
            contract = await current_contract_with_basis(self.client, setup.symbol) if guarded else None
            async with SessionLocal() as session:
                if await session.get(SetupRecord, setup.id) is not None:
                    return False
                active_id = await session.scalar(
                    select(SetupRecord.id)
                    .where(SetupRecord.symbol == setup.symbol, SetupRecord.status == "ACTIVE")
                    .limit(1)
                )
                if active_id is not None:
                    return False

                if not self.settings.auto_execution:
                    # Pending entries reserve the same margin as filled paper positions.
                    # Read persisted reservations under the publication lock so restarts
                    # and simultaneous signals cannot each spend the whole account.
                    rows = (await session.scalars(select(SetupRecord).where(
                        SetupRecord.status == "ACTIVE", SetupRecord.execution_mode == "paper"
                    ))).all()
                    reserved_notional = 0.0
                    reserved_margin = 0.0
                    for row in rows:
                        from app.db.repository import record_to_setup
                        restored = record_to_setup(row)
                        amount = restored.position_size_usdt
                        if amount is None or not math.isfinite(amount) or amount <= 0:
                            self.risk_manager.halted = True
                            raise ValueError("An active paper reservation has no valid position size")
                        reserved_notional += amount
                        margin = getattr(restored, "economics", {}).get(
                            "reserved_margin_usdt", amount / getattr(restored, "leverage", 1))
                        if not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin <= 0:
                            self.risk_manager.halted = True
                            raise ValueError("An active paper reservation has no valid margin")
                        reserved_margin += margin
                    balance = self.risk_manager.balance
                    leverage = self.settings.default_leverage
                    if guarded:
                        # Marked losses cannot be spent again by another paper idea.
                        marked = getattr(self.tracker, "marked_equity", balance)
                        if not isinstance(marked, (int, float)) or not math.isfinite(marked):
                            self.risk_manager.halted = True
                            raise ValueError("Paper marked equity is unavailable")
                        try:
                            setup = prepare_runtime_setup(setup, self.settings, contract, balance,
                                                          max(min(balance, marked) * .9 - reserved_margin, 0.0))
                        except ValueError as exc:
                            self.last_rejections[setup.symbol] = str(exc)
                            return False
                    else:
                        available_margin = max(balance * 0.9 - reserved_notional / leverage, 0.0)
                        risk_size = self.risk_manager.calculate_position_size(
                            setup.entry, setup.stop_loss, self.settings.risk_per_trade_percent, balance
                        )
                        size = min(risk_size, available_margin * leverage)
                        if not math.isfinite(size) or size <= 0:
                            logger.info(f"No paper margin available for {setup.symbol}")
                            return False
                        setup.position_size_usdt = size
                elif guarded:
                    try:
                        setup = prepare_runtime_setup(setup, self.settings, contract, self.risk_manager.balance,
                                                      self.risk_manager.balance * .9)
                    except ValueError as exc:
                        self.last_rejections[setup.symbol] = str(exc)
                        return False

                if guarded:
                    setup.strategy_version = getattr(self.settings, "volium_strategy_profile", "v1_guarded")

            self.risk_manager.open_setups += 1
            saved = False
            try:
                if not self.settings.auto_execution and hasattr(setup, "economics"):
                    # Polling/admission may occur after the signal candle close.
                    # Paper execution starts with a full minute after this time.
                    setup.economics["paper_submitted_at_utc"] = datetime.now(timezone.utc).isoformat()
                async with SessionLocal() as session:
                    await save_setup(session, setup, execution_mode="live" if self.settings.auto_execution else "paper")
                    saved = True
                if self.settings.auto_execution:
                    # False means a definitive rejection. Accepted and uncertain
                    # exchange outcomes retain their reservation for reconciliation.
                    accepted = await self.executor.execute_setup(setup)
                    if not accepted:
                        setup.status = "CANCELLED"
                        async with SessionLocal() as session:
                            await update_setup_status(session, setup.id, "CANCELLED")
                        self.risk_manager.open_setups = max(0, self.risk_manager.open_setups - 1)
                        return False
                await self.tracker.add(setup)
            except Exception:
                if not saved:
                    self.risk_manager.open_setups = max(0, self.risk_manager.open_setups - 1)
                else:
                    # A persisted but interrupted submission remains monitored.
                    await self.tracker.add(setup)
                raise

            self.latest_signals[setup.symbol] = setup
            self.last_rejections.pop(setup.symbol, None)

        if self.ws_manager is not None:
            try:
                await self.ws_manager.broadcast("new_setup", setup.model_dump())
            except Exception as exc:
                logger.warning(f"Setup broadcast failed for {setup.id}: {exc}")
        logger.info(f"New video-strategy setup for {setup.symbol}: {setup.id}")
        return True

    async def _scan_symbol(self, symbol: str, mode: str, now: datetime):
        async with self._fetch_semaphore:
            try:
                timeframes = self._timeframes(mode)
                entry_timeframe = timeframes[-1]
                limit = self.settings.volium_context_lookback + 30
                profile = getattr(self.settings, "volium_strategy_profile", "v1_guarded")
                entry_limit = limit
                if profile.startswith(("v5", "v6")):
                    # Match the preregistered replay prefix: it must include the
                    # liquidity-bar boundary and the correction's FVG history.
                    # One extra bar allows for the currently forming candle,
                    # leaving the replay's full closed prefix available.
                    entry_limit = self.settings.volium_context_lookback * (12 if mode == "intraday" else 5) + 31
                entry_frame = await self.client.get_klines(symbol, entry_timeframe, limit=entry_limit)
                if entry_frame.empty:
                    return
                timestamp = self._latest_timestamp(entry_frame)
                key = (symbol, mode)
                processed = self.last_processed.get(key)
                if processed is not None and timestamp <= processed:
                    return
                context_timeframes = timeframes[:-1]
                context_frames = await asyncio.gather(
                    *(self.client.get_klines(symbol, timeframe, limit=limit) for timeframe in context_timeframes)
                )
                if any(frame.empty for frame in context_frames):
                    return
                frames = dict(zip(context_timeframes, context_frames))
                frames[entry_timeframe] = entry_frame
                if profile == "v6_current_leg":
                    from app.strategy.volium_v6 import analyze_volium_v6_from_df
                    setup = analyze_volium_v6_from_df(symbol=symbol, frames=frames, settings=self.settings, mode=mode, now=now)
                elif profile.startswith("v5"):
                    from app.strategy.volium_v5 import V5Parameters, analyze_volium_v5_from_df
                    setup = analyze_volium_v5_from_df(symbol=symbol, frames=frames, settings=self.settings, mode=mode, now=now,
                        parameters=V5Parameters(liquidity_mode="equal_clusters" if profile == "v5_equal" else "strict"))
                else:
                    setup = analyze_volium_from_df(symbol=symbol, frames=frames, settings=self.settings, mode=mode, now=now)
                self.latest_signals[symbol] = setup
                if setup is not None:
                    await self._publish_setup(setup)
                self.last_processed[key] = timestamp
                self.last_errors.pop(symbol, None)
            except Exception as exc:
                self.last_errors[symbol] = str(exc)
                logger.exception(f"Video-strategy scan failed for {symbol}: {exc}")

    async def scan(self):
        async with self._scan_lock:
            now = datetime.now(timezone.utc)
            mode = self.settings.volium_mode
            if self.settings.pair_selection == "dynamic":
                try:
                    self.selected_pairs = await self.universe.select()
                    if len(self.selected_pairs) != self.settings.pair_count:
                        raise ValueError("Five valid large-cap pairs are required")
                    self.active_symbols = [p["symbol"] for p in self.selected_pairs]
                    self.selection_error = None
                except Exception:
                    self.selection_error = "Pair selection unavailable; new signal scanning paused"
                    self.last_scan_at = now
                    return
            else:
                self.active_symbols = self.settings.trading_symbols
            symbols = list(dict.fromkeys(self.active_symbols))
            await asyncio.gather(*(self._scan_symbol(symbol, mode, now) for symbol in symbols))
            self.last_scan_at = datetime.now(timezone.utc)

    async def run(self):
        while True:
            await self.scan()
            await asyncio.sleep(self.settings.scan_interval_seconds)


PairScanner = SignalScanner
