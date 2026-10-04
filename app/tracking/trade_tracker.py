from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import pandas as pd
from loguru import logger
from sqlalchemy import select, update
from app.config import Settings
from app.db.engine import SessionLocal
from app.db.models import SetupRecord, ExecutionRecord
from app.db.repository import record_to_setup, update_setup_status
from app.execution.executor import HALT_STATES, as_utc
from app.schemas.setup import TradeSetup


class TradeTracker:
    PAPER_BACKFILL_LIMIT = 100_000
    def __init__(self, client, settings: Settings, risk_manager, executor, ws_manager=None):
        self.client = client
        self.settings = settings
        self.risk_manager = risk_manager
        self.executor = executor
        self.ws = ws_manager
        self.active: dict[str, TradeSetup] = {}
        self.filled_at: dict[str, datetime] = {}
        self.last_bar: dict[str, datetime] = {}
        self._close_lock = asyncio.Lock()

    async def restore(self):
        self.active.clear()
        self.filled_at.clear()
        self.last_bar.clear()
        mode = "live" if self.settings.auto_execution else "paper"
        async with SessionLocal() as session:
            rows = (await session.scalars(select(SetupRecord).where(SetupRecord.status == "ACTIVE"))).all()
            for row in rows:
                if row.execution_mode != mode:
                    if row.execution_mode == "live":
                        self.risk_manager.halted = True
                    continue
                setup = record_to_setup(row)
                self.active[setup.id] = setup
                if row.paper_filled_at:
                    self.filled_at[setup.id] = as_utc(row.paper_filled_at)
                if row.paper_last_bar_at:
                    self.last_bar[setup.id] = as_utc(row.paper_last_bar_at)
            self.risk_manager.open_setups = len(rows)
            today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            closed = (await session.scalars(select(SetupRecord).where(
                SetupRecord.closed_at >= today, SetupRecord.closed_at < today + timedelta(days=1),
                SetupRecord.execution_mode == mode))).all()
            self.risk_manager.day = today.date()
            self.risk_manager.daily_pnl = sum(row.pnl_usdt or 0 for row in closed)
            if mode == "paper":
                all_closed = (await session.scalars(select(SetupRecord).where(
                    SetupRecord.execution_mode == "paper", SetupRecord.pnl_usdt != None))).all()
                self.risk_manager.balance = self.settings.account_balance_usdt + sum(row.pnl_usdt or 0 for row in all_closed)
                self.risk_manager.daily_opening_balance = self.risk_manager.balance - self.risk_manager.daily_pnl
                self.risk_manager.daily_opening_balance_source = "restored_paper_ledger"
            uncertain = await session.scalar(select(ExecutionRecord).where(ExecutionRecord.state.in_(HALT_STATES)).limit(1))
            if uncertain:
                self.risk_manager.halted = True
        # A saved live signal with no journal was never safely submitted.
        if self.settings.auto_execution:
            async with SessionLocal() as session:
                for setup in list(self.active.values()):
                    if await session.get(ExecutionRecord, setup.id) is None:
                        await update_setup_status(session, setup.id, "CANCELLED")
                        self.active.pop(setup.id)
                        self.risk_manager.open_setups = max(0, self.risk_manager.open_setups-1)

    async def add(self, setup: TradeSetup):
        self.active[setup.id] = setup

    async def _close_setup(self, setup, status, pnl_usdt=None, *, closed_at=None):
        """Paper uses the observed bar close; live falls back to reconciliation time.

        Without an authoritative live closing timestamp the exchange confirmation
        time is recorded, which can differ from the execution's actual UTC day.
        """
        closed_at = as_utc(closed_at or datetime.now(timezone.utc))
        async with self._close_lock:
            async with SessionLocal() as session:
                result = await session.execute(
                    update(SetupRecord)
                    .where(SetupRecord.id == setup.id, SetupRecord.status == "ACTIVE")
                    .values(status=status, pnl_usdt=pnl_usdt, closed_at=closed_at)
                )
                await session.commit()
                changed = bool(result.rowcount)
                record = await session.get(SetupRecord, setup.id)
                if record is None:
                    raise RuntimeError("Cannot close a setup without its persisted reservation")
                setup.status = record.status
            if changed:
                if pnl_usdt is not None:
                    self.risk_manager.record_result(pnl_usdt, closed_at=closed_at)
                    if not self.settings.auto_execution:
                        self.risk_manager.balance += pnl_usdt
                self.risk_manager.open_setups = max(0, self.risk_manager.open_setups-1)
            self.active.pop(setup.id, None)
            self.filled_at.pop(setup.id, None)
            self.last_bar.pop(setup.id, None)
        if changed and self.ws:
            try:
                await self.ws.broadcast("setup_update", setup.model_dump(mode="json"))
            except Exception:
                logger.warning("Setup update broadcast failed for {}", setup.id)

    async def process_paper_bars(self, symbol, frame):
        for setup in list(self.active.values()):
            if setup.symbol != symbol:
                continue
            cutoff = self.last_bar.get(setup.id, as_utc(setup.timestamp) - timedelta(microseconds=1))
            for ts, row in frame.iterrows():
                bar_at = as_utc(ts.to_pydatetime())
                if bar_at <= cutoff or bar_at < as_utc(setup.timestamp):
                    continue
                high, low = float(row["high"]), float(row["low"])
                filled_before_bar = setup.id in self.filled_at
                marketable_at_open = float(row["open"]) <= setup.entry if setup.direction == "LONG" else float(row["open"]) >= setup.entry
                if setup.id not in self.filled_at:
                    if bar_at >= as_utc(setup.timestamp) + timedelta(minutes=self.settings.pending_order_max_age_minutes):
                        await self._close_setup(setup, "EXPIRED", closed_at=bar_at + timedelta(minutes=1))
                        break
                    if low <= setup.entry <= high:
                        self.filled_at[setup.id] = bar_at
                    elif (setup.direction == "LONG" and high < setup.entry) or (setup.direction == "SHORT" and low > setup.entry):
                        # A marketable limit can fill at its price cap in a gapped bar.
                        self.filled_at[setup.id] = bar_at
                if setup.id in self.filled_at:
                    sl = low <= setup.stop_loss if setup.direction == "LONG" else high >= setup.stop_loss
                    tp = high >= setup.take_profits[0] if setup.direction == "LONG" else low <= setup.take_profits[0]
                    if not filled_before_bar and not marketable_at_open:
                        # OHLC cannot prove a favorable target touch followed the entry fill.
                        tp = False
                    if sl or tp:
                        # Unknown intrabar order: stop wins whenever both levels are touched.
                        exit_price = setup.stop_loss if sl else setup.take_profits[0]
                        # Adverse gaps through stop are filled at the bar's open.
                        if sl:
                            exit_price = min(exit_price, float(row["open"])) if setup.direction == "LONG" else max(exit_price, float(row["open"]))
                        sign = 1 if setup.direction == "LONG" else -1
                        gross = sign * (exit_price-setup.entry)/setup.entry
                        costs = 2*(self.settings.paper_fee_bps+self.settings.paper_slippage_bps)/10000
                        pnl = (setup.position_size_usdt or 0)*(gross-costs)
                        await self._close_setup(setup, "SL_HIT" if sl else "TP1_HIT", pnl,
                            closed_at=bar_at + timedelta(minutes=1))
                        break
                self.last_bar[setup.id] = bar_at
                async with SessionLocal() as session:
                    rec = await session.get(SetupRecord, setup.id)
                    if rec:
                        rec.paper_filled_at = self.filled_at.get(setup.id)
                        rec.paper_last_bar_at = bar_at
                        await session.commit()

    async def cancel(self, setup_id):
        setup = self.active.get(setup_id)
        if setup is None:
            raise ValueError("Active setup not found")
        if self.settings.auto_execution:
            await self.executor.cancel_pending(setup_id)
            return {"status": "cancellation_requested", "exchange_confirmation_pending": True}
        if setup_id in self.filled_at:
            raise ValueError("Paper position is filled; close it using its bracket or a separate close action")
        await self._close_setup(setup, "CANCELLED")
        return {"status": "CANCELLED"}

    async def _paper_backfill(self, symbol, closed_at_ms):
        minute_ms = 60_000
        latest_open_ms = (closed_at_ms // minute_ms - 1) * minute_ms
        starts = []
        for setup in self.active.values():
            if setup.symbol != symbol:
                continue
            previous = self.last_bar.get(setup.id)
            if previous is not None:
                starts.append(int(previous.timestamp() * 1000) + minute_ms)
            else:
                earliest = min(as_utc(setup.timestamp), self.filled_at.get(setup.id, as_utc(setup.timestamp)))
                starts.append(int(pd.Timestamp(earliest).ceil("min").timestamp() * 1000))
        if not starts or min(starts) > latest_open_ms:
            return
        start_ms = min(starts)
        count = (latest_open_ms - start_ms) // minute_ms + 1
        if count > self.PAPER_BACKFILL_LIMIT:
            raise ValueError("Paper history exceeds the recoverable candle range; entries remain paused")
        frame = await self.client.get_klines(
            symbol, "1m", limit=count, start_time=start_ms, end_time=closed_at_ms
        )
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("Paper history has an invalid time index")
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame.loc[(frame.index >= pd.Timestamp(start_ms, unit="ms", tz="UTC")) &
                          (frame.index <= pd.Timestamp(latest_open_ms, unit="ms", tz="UTC"))].sort_index()
        expected = pd.date_range(pd.Timestamp(start_ms, unit="ms", tz="UTC"), periods=count, freq="min")
        if not frame.index.equals(expected):
            # Validate the complete gap before advancing any setup's durable cursor.
            raise ValueError("Paper candle history is incomplete; entries remain paused")
        await self.process_paper_bars(symbol, frame)

    async def poll(self):
        if self.settings.auto_execution:
            for setup in list(self.active.values()):
                result = await self.executor.reconcile(setup)
                if result:
                    status, _, pnl = result
                    await self._close_setup(setup, status, pnl)
            balance = await self.client.get_balance()
            self.risk_manager.balance = float(balance["equity"])
        else:
            symbols = {s.symbol for s in self.active.values()}
            if symbols:
                try:
                    closed_at_ms = int(await self.client.get_server_time()) - 1500
                    for symbol in symbols:
                        await self._paper_backfill(symbol, closed_at_ms)
                except Exception:
                    self.risk_manager.halted = True
                    raise
        async with SessionLocal() as session:
            unresolved = await session.scalar(select(ExecutionRecord).where(ExecutionRecord.state.in_(HALT_STATES)).limit(1))
            inactive_live = None
            if not self.settings.auto_execution:
                inactive_live = await session.scalar(select(SetupRecord.id).where(
                    SetupRecord.execution_mode == "live", SetupRecord.status == "ACTIVE").limit(1))
            self.risk_manager.halted = bool(unresolved or inactive_live)

    async def track_loop(self):
        while True:
            try:
                await self.poll()
            except Exception:
                logger.warning("Trade reconciliation failed; entries paused until next successful poll")
                self.risk_manager.halted = True
            await asyncio.sleep(self.settings.scan_interval_seconds)
