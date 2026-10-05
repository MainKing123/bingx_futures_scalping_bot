from __future__ import annotations

import asyncio
import copy
import json
import math
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
        self._marks: dict[str, float] = {}
        self._funding_ok_symbols: set[str] = set()
        self._close_lock = asyncio.Lock()
        self._paper_lock = asyncio.Lock()

    @staticmethod
    def _runtime_paper(setup):
        from app.execution.economics import RuntimeTradeSetup
        return isinstance(setup, RuntimeTradeSetup)

    @staticmethod
    def _runtime_activation(setup):
        """Paper admission can occur after the strategy candle has closed."""
        signal_at = as_utc(setup.timestamp)
        if "paper_submitted_at_utc" not in setup.economics:
            # Compatibility for old/manual records with no admission timestamp.
            return signal_at
        stamp = setup.economics["paper_submitted_at_utc"]
        if not isinstance(stamp, str):
            raise ValueError("Invalid paper submission timestamp")
        try:
            submitted = datetime.fromisoformat(stamp)
        except ValueError:
            raise ValueError("Invalid paper submission timestamp") from None
        if submitted.tzinfo is None:
            raise ValueError("Paper submission timestamp must include its timezone")
        return max(signal_at, as_utc(submitted))

    @classmethod
    def _runtime_first_bar(cls, setup):
        # A partial admission minute cannot establish post-admission touches.
        return pd.Timestamp(cls._runtime_activation(setup)).ceil("min").to_pydatetime()

    @staticmethod
    def _finite(value, name, *, positive=False):
        if isinstance(value, bool):
            raise ValueError("Invalid paper " + name)
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError("Invalid paper " + name) from None
        if not math.isfinite(number) or (positive and number <= 0):
            raise ValueError("Invalid paper " + name)
        return number

    @property
    def marked_equity(self):
        """Cash plus unrealized PnL at observed closed M1 prices, not fair price."""
        equity = self._finite(self.risk_manager.balance, "cash")
        for identifier, setup in self.active.items():
            if identifier not in self.filled_at:
                continue
            mark = self._marks.get(identifier)
            if mark is None:
                self.risk_manager.halted = True
                raise ValueError("Filled paper position has no observed price; entries remain paused")
            notional = self._finite(setup.position_size_usdt, "notional", positive=True)
            sign = 1 if setup.direction == "LONG" else -1
            equity += sign * notional * (self._finite(mark, "mark", positive=True) - setup.entry) / setup.entry
        return equity

    def _restore_paper_cash(self, rows, today):
        """Rebuild absolute cash from durable events, never apply them twice."""
        total, daily = 0.0, 0.0
        for row in rows:
            setup = record_to_setup(row)
            ledger = getattr(setup, "economics", {}).get("paper_ledger")
            if ledger is not None:
                if ledger.get("version") != 1 or not isinstance(ledger.get("events"), list):
                    raise ValueError("Invalid durable paper cash ledger")
                events = ledger["events"]
                if len({event["id"] for event in events}) != len(events):
                    raise ValueError("Duplicate durable paper cash event")
                subtotal = 0.0
                for event in events:
                    delta = self._finite(event["cash_delta_usdt"], "cash event")
                    event_at = as_utc(datetime.fromisoformat(event["at"]))
                    subtotal += delta
                    if today <= event_at < today + timedelta(days=1):
                        daily += delta
                if row.pnl_usdt is not None and not math.isclose(subtotal, row.pnl_usdt, abs_tol=1e-8):
                    raise ValueError("Closed paper trade and cash ledger disagree")
                total += subtotal
                if row.status == "ACTIVE" and row.paper_filled_at:
                    if not any(event["id"] == "entry" for event in events):
                        raise ValueError("Filled paper position has no durable entry cost")
                    self._marks[setup.id] = self._finite(ledger.get("last_mark"), "mark", positive=True)
            elif self._runtime_paper(setup) and row.paper_filled_at:
                raise ValueError("Filled runtime paper position has no durable cash ledger; entries remain paused")
            elif row.pnl_usdt is not None:
                # Frozen legacy rows continue to book their total PnL at closing.
                delta = self._finite(row.pnl_usdt, "legacy result")
                total += delta
                if row.closed_at and today <= as_utc(row.closed_at) < today + timedelta(days=1):
                    daily += delta
        self.risk_manager.daily_pnl = daily
        self.risk_manager.balance = self.settings.account_balance_usdt + total
        self.risk_manager.daily_opening_balance = self.risk_manager.balance - daily
        self.risk_manager.daily_opening_balance_source = "restored_paper_ledger"

    async def restore(self):
        self.active.clear()
        self.filled_at.clear()
        self.last_bar.clear()
        self._marks.clear()
        self._funding_ok_symbols.clear()
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
                if mode == "paper" and self._runtime_paper(setup):
                    # A restored wallet cannot admit entries before public
                    # candles and funding coverage have been reconciled.
                    self.risk_manager.halted = True
                    first_bar = self._runtime_first_bar(setup)
                    if row.paper_filled_at and as_utc(row.paper_filled_at) < first_bar:
                        raise ValueError("Paper fill precedes the actual submission; entries remain paused")
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
                paper_rows = (await session.scalars(select(SetupRecord).where(
                    SetupRecord.execution_mode == "paper"))).all()
                try:
                    self._restore_paper_cash(paper_rows, today)
                except Exception:
                    self.risk_manager.halted = True
                    raise
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
            self._marks.pop(setup.id, None)
        if changed and self.ws:
            try:
                await self.ws.broadcast("setup_update", setup.model_dump(mode="json"))
            except Exception:
                logger.warning("Setup update broadcast failed for {}", setup.id)

    async def process_paper_bars(self, symbol, frame):
        # Serialize validation, durable cash commits and in-memory booking.
        async with self._paper_lock:
            try:
                await self._process_paper_bars(symbol, frame)
            except Exception:
                self.risk_manager.halted = True
                raise

    async def _process_paper_bars(self, symbol, frame):
        runtime = [setup for setup in self.active.values() if setup.symbol == symbol and self._runtime_paper(setup)]
        funding = None
        if runtime and not frame.empty:
            if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
                raise ValueError("Runtime paper candles require unique ascending aware timestamps")
            needs_processing = [setup for setup in runtime if frame.index[-1].to_pydatetime() >
                self.last_bar.get(setup.id, self._runtime_first_bar(setup) - timedelta(microseconds=1))]
            if needs_processing:
                for setup in needs_processing:
                    previous = self.last_bar.get(setup.id)
                    start = pd.Timestamp(previous + timedelta(minutes=1)) if previous else pd.Timestamp(self._runtime_first_bar(setup))
                    new = frame.loc[frame.index >= start]
                    expected = pd.date_range(start, frame.index[-1], freq="min")
                    if not new.index.equals(expected):
                        raise ValueError("Runtime paper candle history is incomplete or does not begin at the pending order")
                    for _, candle in new.iterrows():
                        prices = {key: self._finite(candle[key], key, positive=True) for key in ("open", "high", "low", "close")}
                        if not prices["low"] <= min(prices["open"], prices["close"]) <= max(prices["open"], prices["close"]) <= prices["high"]:
                            raise ValueError("Invalid paper candle OHLC geometry")
                starts = [max(self._runtime_first_bar(setup), self.last_bar.get(setup.id, self._runtime_first_bar(setup))) for setup in needs_processing]
                funding = await self._validated_public_funding(symbol, min(starts))
        for setup in list(self.active.values()):
            if setup.symbol != symbol:
                continue
            first_bar = self._runtime_first_bar(setup) if self._runtime_paper(setup) else as_utc(setup.timestamp)
            cutoff = self.last_bar.get(setup.id, first_bar - timedelta(microseconds=1))
            for ts, row in frame.iterrows():
                bar_at = as_utc(ts.to_pydatetime())
                if bar_at <= cutoff or bar_at < first_bar:
                    continue
                if self._runtime_paper(setup):
                    await self._process_runtime_bar(setup, bar_at, row, funding)
                    if setup.id not in self.active:
                        break
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
                        costs = getattr(setup, "economics", {}).get("roundtrip_cost_fraction",
                            2*(self.settings.paper_fee_bps+self.settings.paper_slippage_bps)/10000)
                        pnl = (setup.position_size_usdt or 0)*(gross-costs)
                        await self._close_setup(setup, "SL_HIT" if sl else "TP1_HIT", pnl,
                            closed_at=bar_at + timedelta(minutes=1))
                        break
                self.last_bar[setup.id] = bar_at
                self._marks[setup.id] = float(row["close"])
                async with SessionLocal() as session:
                    rec = await session.get(SetupRecord, setup.id)
                    if rec:
                        rec.paper_filled_at = self.filled_at.get(setup.id)
                        rec.paper_last_bar_at = bar_at
                        await session.commit()

    async def _validated_public_funding(self, symbol, start):
        """Validate actual rates and settlement coverage before any candle commits.

        The frozen client's Series omits historical collectCycle, so a separate
        public history read audits schedules. No private account endpoint is used.
        """
        since_ms = int((start - timedelta(hours=48)).timestamp() * 1000)
        rates = await self.client.get_funding_history(symbol, since_ms)
        if not isinstance(rates, pd.Series) or not isinstance(rates.index, pd.DatetimeIndex) or rates.index.tz is None or not rates.index.is_unique or not rates.index.is_monotonic_increasing:
            raise ValueError("Public funding history has an invalid schema")
        rates = rates.copy()
        rates.index = pd.to_datetime(rates.index, utc=True)
        for rate in rates:
            self._finite(rate, "funding rate")
        schedule = await self.client._request("GET", f"/api/v1/contract/funding_rate/{symbol}", signed=False)
        if not isinstance(schedule, dict) or schedule.get("symbol") != symbol:
            raise ValueError("Public funding schedule belongs to another contract")
        observed = self._finite(schedule.get("timestamp"), "funding observation", positive=True)
        age = datetime.now(timezone.utc).timestamp() - observed / 1000
        if not -5 <= age <= 90:
            raise ValueError("Public funding schedule is stale")
        cycle = self._funding_cycle(schedule.get("collectCycle"))
        next_ms = self._finite(schedule.get("nextSettleTime"), "next settlement", positive=True)
        if next_ms <= observed or next_ms - observed > cycle * 3_600_000:
            raise ValueError("Public funding next settlement is invalid")
        published = {}
        for page in range(1, 101):
            raw = await self.client._request("GET", "/api/v1/contract/funding_rate/history",
                {"symbol": symbol, "page_num": page, "page_size": 1000}, signed=False)
            rows = raw.get("resultList") if isinstance(raw, dict) else None
            if not isinstance(rows, list):
                raise ValueError("Public funding cycle history is unavailable")
            if not rows:
                break
            stamps = []
            for record in rows:
                if record.get("symbol") != symbol:
                    raise ValueError("Public funding history belongs to another contract")
                stamp = self._finite(record.get("settleTime"), "settlement timestamp", positive=True)
                if not stamp.is_integer():
                    raise ValueError("Invalid settlement timestamp")
                stamp = int(stamp)
                stamps.append(stamp)
                item = (self._finite(record.get("fundingRate"), "funding rate"),
                        self._funding_cycle(record.get("collectCycle")))
                if stamp in published and published[stamp] != item:
                    raise ValueError("Conflicting public funding settlement")
                published[stamp] = item
            if min(stamps) <= since_ms or len(rows) < 1000:
                break
        else:
            raise ValueError("Public funding audit exceeded its pagination limit")
        usable = sorted(stamp for stamp in published if stamp >= since_ms)
        actual = {int(stamp.timestamp() * 1000): float(rate) for stamp, rate in rates.items()}
        if set(actual) != set(usable) or any(actual[stamp] != published[stamp][0] for stamp in usable):
            raise ValueError("Public funding rate and cycle snapshots disagree")
        if not usable or usable[0] > int(start.timestamp() * 1000):
            raise ValueError("Public funding history has no preceding settlement")
        for before, after in zip(usable, usable[1:]):
            if after - before != published[after][1] * 3_600_000:
                raise ValueError("Public funding settlement history is incomplete or its schedule cannot be verified")
        if usable[-1] > observed or usable[-1] + cycle * 3_600_000 != next_ms:
            raise ValueError("Public funding newest settlement is missing or its schedule cannot be verified")
        rates.attrs["observed_at"] = datetime.fromtimestamp(observed / 1000, timezone.utc)
        self._funding_ok_symbols.add(symbol)
        return rates

    @classmethod
    def _funding_cycle(cls, value):
        cycle = cls._finite(value, "funding cycle", positive=True)
        if not cycle.is_integer() or cycle > 24:
            raise ValueError("Unsupported published funding cycle")
        return int(cycle)

    async def _process_runtime_bar(self, setup, bar_at, row, funding):
        """Persist one complete paper cash step before updating in-memory risk."""
        if funding is None:
            raise ValueError("Runtime paper funding coverage was not validated")
        prices = {key: self._finite(row[key], key, positive=True) for key in ("open", "high", "low", "close")}
        if not prices["low"] <= min(prices["open"], prices["close"]) <= max(prices["open"], prices["close"]) <= prices["high"]:
            raise ValueError("Invalid paper candle OHLC geometry")
        previous = self.last_bar.get(setup.id)
        if previous is not None and bar_at != previous + timedelta(minutes=1):
            raise ValueError("Runtime paper candle history is incomplete")
        if previous is None and bar_at != self._runtime_first_bar(setup):
            raise ValueError("Runtime paper candle history does not begin at the pending order")
        if bar_at + timedelta(minutes=1) > funding.attrs["observed_at"]:
            raise ValueError("Runtime paper candle is not closed at the funding observation")
        notional = self._finite(setup.position_size_usdt, "notional", positive=True)
        fee_rate = self._finite(setup.economics.get("modeled_fee_bps_per_side"), "modeled fee") / 10000
        slip_rate = self._finite(setup.economics.get("modeled_slippage_bps_per_side"), "modeled slippage") / 10000
        if fee_rate < 0 or slip_rate < 0:
            raise ValueError("Negative paper trading costs")
        old = setup.economics.get("paper_ledger")
        if old is None and setup.id in self.filled_at:
            raise ValueError("Filled runtime paper position has no durable cash ledger")
        ledger = copy.deepcopy(old or {"version": 1, "events": []})
        if ledger.get("version") != 1 or not isinstance(ledger.get("events"), list):
            raise ValueError("Invalid durable paper cash ledger")
        before_count = len(ledger["events"])
        bar_close = bar_at + timedelta(minutes=1)
        filled_before = setup.id in self.filled_at
        filled_at = self.filled_at.get(setup.id)
        market_open = prices["open"] <= setup.entry if setup.direction == "LONG" else prices["open"] >= setup.entry
        status, pnl = "ACTIVE", None
        if filled_at is None:
            expiry = as_utc(setup.timestamp) + timedelta(minutes=self.settings.pending_order_max_age_minutes)
            touched = prices["low"] <= setup.entry <= prices["high"]
            if bar_at >= expiry or (bar_close > expiry and not market_open):
                status = "EXPIRED"
            elif market_open or touched:
                filled_at = bar_at
                ledger["events"].append({"id": "entry", "at": bar_at.isoformat(),
                    "cash_delta_usdt": -notional * (fee_rate + slip_rate),
                    "fee_usdt": notional * fee_rate, "slippage_usdt": notional * slip_rate})
        if filled_at is not None:
            long = setup.direction == "LONG"
            sl = prices["low"] <= setup.stop_loss if long else prices["high"] >= setup.stop_loss
            tp = prices["high"] >= setup.take_profits[0] if long else prices["low"] <= setup.take_profits[0]
            if not filled_before and not market_open:
                tp = False
            sign = 1 if long else -1
            exit_open = ((sl and (prices["open"] <= setup.stop_loss if long else prices["open"] >= setup.stop_loss))
                or (tp and (prices["open"] >= setup.take_profits[0] if long else prices["open"] <= setup.take_profits[0])))
            consumed = {event["id"] for event in ledger["events"]}
            for stamp, rate in funding.loc[(funding.index >= pd.Timestamp(bar_at)) & (funding.index <= pd.Timestamp(bar_close))].items():
                settlement = stamp.to_pydatetime()
                event_id = "funding:" + str(int(stamp.timestamp() * 1000))
                if event_id in consumed or settlement < filled_at or (exit_open and settlement > bar_at):
                    continue
                # The public endpoint has no historical settlement fair price.
                # Use an observed closed candle; at the close use this candle.
                mark = prices["close"] if settlement == bar_close else ledger.get("last_mark", prices["open"])
                mark = self._finite(mark, "funding price proxy", positive=True)
                cash = -sign * float(rate) * notional / setup.entry * mark
                known_held = not (sl or tp) and (filled_before or (market_open and settlement > bar_at))
                skipped = cash > 0 and not known_held
                ledger["events"].append({"id": event_id, "at": settlement.isoformat(),
                    "cash_delta_usdt": 0.0 if skipped else cash, "settlement_rate": float(rate),
                    "observed_price_proxy": mark, "historical_fair_price_available": False,
                    "favorable_skipped_intrabar_uncertainty": skipped})
            if sl or tp:
                exit_price = setup.stop_loss if sl else setup.take_profits[0]
                if sl:
                    exit_price = min(exit_price, prices["open"]) if long else max(exit_price, prices["open"])
                gross = sign * notional * (exit_price - setup.entry) / setup.entry
                exit_notional = notional / setup.entry * exit_price
                ledger["events"].append({"id": "exit", "at": bar_close.isoformat(),
                    "cash_delta_usdt": gross - exit_notional * (fee_rate + slip_rate),
                    "gross_price_pnl_usdt": gross, "exit_price": exit_price,
                    "fee_usdt": exit_notional * fee_rate, "slippage_usdt": exit_notional * slip_rate})
                status = "SL_HIT" if sl else "TP1_HIT"
                pnl = sum(event["cash_delta_usdt"] for event in ledger["events"])
        ledger.update(last_mark=prices["close"], last_mark_at=bar_close.isoformat(),
                      funding_checked_through=bar_close.isoformat())
        economics = {**setup.economics, "paper_ledger": ledger}
        if filled_at is not None:
            economics["reserved_margin_usdt"] = notional / setup.leverage
        payload = setup.model_copy(update={"economics": economics, "status": status}).model_dump_json()
        async with self._close_lock:
            async with SessionLocal() as session:
                rec = await session.get(SetupRecord, setup.id)
                if rec is None or rec.status != "ACTIVE":
                    raise RuntimeError("Cannot book paper cash without an active persisted reservation")
                saved_previous = as_utc(rec.paper_last_bar_at) if rec.paper_last_bar_at else None
                if saved_previous != previous:
                    raise RuntimeError("Paper cash cursor changed in another worker; restore before retrying")
                saved_ledger = json.loads(rec.payload).get("economics", {}).get("paper_ledger", {"events": []})
                if len(saved_ledger["events"]) != before_count:
                    raise RuntimeError("Paper cash ledger changed in another worker; restore before retrying")
                values = {"payload": payload, "paper_filled_at": filled_at, "paper_last_bar_at": bar_at}
                if status != "ACTIVE":
                    values.update(status=status, pnl_usdt=pnl, closed_at=bar_close)
                predicate = SetupRecord.paper_last_bar_at == previous if previous else SetupRecord.paper_last_bar_at.is_(None)
                result = await session.execute(update(SetupRecord).where(SetupRecord.id == setup.id,
                    SetupRecord.status == "ACTIVE", predicate).values(**values))
                if result.rowcount != 1:
                    raise RuntimeError("Paper cash reservation changed; restore before retrying")
                await session.commit()
            setup.economics, setup.status = economics, status
            for event in ledger["events"][before_count:]:
                delta = event["cash_delta_usdt"]
                self.risk_manager.record_result(delta, closed_at=as_utc(datetime.fromisoformat(event["at"])))
                self.risk_manager.balance += delta
            self.last_bar[setup.id] = bar_at
            self._marks[setup.id] = prices["close"]
            if filled_at is not None:
                self.filled_at[setup.id] = filled_at
            if status != "ACTIVE":
                self.risk_manager.open_setups = max(0, self.risk_manager.open_setups - 1)
                self.active.pop(setup.id, None)
                self.filled_at.pop(setup.id, None)
                self.last_bar.pop(setup.id, None)
                self._marks.pop(setup.id, None)
        if status != "ACTIVE" and self.ws:
            try:
                await self.ws.broadcast("setup_update", setup.model_dump(mode="json"))
            except Exception:
                logger.warning("Setup update broadcast failed for {}", setup.id)

    async def cancel(self, setup_id):
        setup = self.active.get(setup_id)
        if setup is None:
            raise ValueError("Active setup not found")
        if self.settings.auto_execution:
            await self.executor.cancel_pending(setup_id)
            return {"status": "cancellation_requested", "exchange_confirmation_pending": True}
        async with self._paper_lock:
            # Recheck after a concurrent candle has committed its fill state.
            setup = self.active.get(setup_id)
            if setup is None:
                raise ValueError("Active setup not found")
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
                if self._runtime_paper(setup):
                    earliest = self._runtime_first_bar(setup)
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
            self._funding_ok_symbols.clear()
            symbols = {s.symbol for s in self.active.values()}
            if symbols:
                try:
                    closed_at_ms = int(await self.client.get_server_time()) - 1500
                    for symbol in symbols:
                        await self._paper_backfill(symbol, closed_at_ms)
                        runtime = [setup for setup in self.active.values()
                            if setup.symbol == symbol and self._runtime_paper(setup)]
                        if runtime and symbol not in self._funding_ok_symbols:
                            start = min(self.last_bar.get(setup.id, self._runtime_first_bar(setup)) for setup in runtime)
                            await self._validated_public_funding(symbol, start)
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
