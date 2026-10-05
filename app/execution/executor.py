from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import select

from app.config import Settings
from app.db.engine import SessionLocal
from app.db.models import ExecutionRecord
from app.exchange.client import MEXCClient, MEXCAPIError
from app.schemas.setup import TradeSetup
from app.execution.economics import RuntimeTradeSetup, prepare_runtime_setup, current_contract_with_basis

RESERVED_STATES = {"PREPARED", "UNKNOWN", "ACCEPTED", "FILLED", "UNPROTECTED"}
HALT_STATES = {"PREPARED", "UNKNOWN", "UNPROTECTED"}


def utcnow():
    return datetime.now(timezone.utc)


def as_utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class AutoExecutor:
    """One durable attempt per signal; the exchange confirms fills and closes."""
    def __init__(self, client: MEXCClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.lock = asyncio.Lock()

    async def _update(self, setup_id, **fields):
        async with SessionLocal() as session:
            record = await session.get(ExecutionRecord, setup_id)
            if record is None:
                raise RuntimeError("Execution reservation missing")
            for key, value in fields.items():
                setattr(record, key, value)
            record.updated_at = utcnow()
            await session.commit()

    async def execute_setup(self, setup: TradeSetup) -> bool:
        if not self.settings.auto_execution:
            return False
        async with self.lock:
            async with SessionLocal() as session:
                previous = await session.get(ExecutionRecord, setup.id)
                if previous:
                    return previous.state in RESERVED_STATES
                halt = await session.scalar(select(ExecutionRecord).where(ExecutionRecord.state.in_(HALT_STATES)).limit(1))
                if halt:
                    logger.error("New entries paused: an execution requires reconciliation")
                    return False
                symbol_reserved = await session.scalar(select(ExecutionRecord).where(
                    ExecutionRecord.symbol == setup.symbol, ExecutionRecord.state.in_(RESERVED_STATES)).limit(1))
                if symbol_reserved:
                    return False
                external_oid = "vx" + hashlib.sha256(setup.id.encode()).hexdigest()[:28]
                record = ExecutionRecord(setup_id=setup.id, external_oid=external_oid,
                    symbol=setup.symbol, direction=setup.direction, state="PREPARED",
                    created_at=utcnow(), updated_at=utcnow())
                session.add(record)
                await session.commit()  # committed before any network mutation
            try:
                if not setup.take_profits:
                    raise ValueError("Missing take profit")
                tp = setup.take_profits[0]
                if not (setup.stop_loss < setup.entry < tp if setup.direction == "LONG" else tp < setup.entry < setup.stop_loss):
                    raise ValueError("Invalid entry/stop/target ordering")
                positions = await self.client.get_positions(setup.symbol)
                if any(float(p.get("holdVol", 0)) > 0 for p in positions):
                    raise ValueError("An exchange position already exists for symbol")
                for page in range(1, 51):
                    orders = await self.client.get_open_orders(page_num=page, page_size=100)
                    if any(o.get("symbol") == setup.symbol for o in orders):
                        raise ValueError("An exchange order already exists for symbol")
                    if len(orders) < 100:
                        break
                else:
                    raise ValueError("Too many exchange orders to reconcile safely")
                balance = await self.client.get_balance()
                equity = float(balance.get("equity", 0))
                available = float(balance.get("availableBalance", 0))
                if equity <= 0 or available <= 0:
                    raise ValueError("No available USDT equity/margin")
                guarded = getattr(self.settings, "execution_cost_guard_enabled", False)
                if guarded:
                    contract = await current_contract_with_basis(self.client, setup.symbol)
                    prepared = prepare_runtime_setup(setup, self.settings, contract, equity, available * .9,
                                                     preferred_cap=getattr(setup, "leverage", None))
                    # Update the durable setup with rounded geometry, chosen
                    # leverage and fee-inclusive cash risk before submission.
                    if isinstance(setup, RuntimeTradeSetup):
                        for field in type(prepared).model_fields:
                            setattr(setup, field, getattr(prepared, field))
                    else:
                        setup = prepared
                    tp = setup.take_profits[0]
                    notional = setup.position_size_usdt
                    from app.db.models import SetupRecord
                    async with SessionLocal() as session:
                        saved = await session.get(SetupRecord, setup.id)
                        if saved:
                            saved.payload = setup.model_dump_json()
                            await session.commit()
                else:
                    distance = abs(setup.entry - setup.stop_loss) / setup.entry
                    risk_notional = equity * self.settings.risk_per_trade_percent / 100 / distance
                    notional = min(risk_notional, available * self.settings.default_leverage * 0.9)
                setup.position_size_usdt = notional
            except Exception:
                # No create request has been attempted, so rejection is unambiguous.
                await self._update(setup.id, state="REJECTED", error="Preflight rejected; inspect account and setup")
                return False
            response_received = False
            try:
                result = await self.client.place_bracket_order(symbol=setup.symbol, direction=setup.direction,
                    notional_usdt=notional, entry=setup.entry, stop_loss=setup.stop_loss,
                    take_profit=tp, leverage=getattr(setup, "leverage", self.settings.default_leverage), external_oid=external_oid)
                response_received = True
                order_id = result.get("orderId")
                if not order_id:
                    await self._update(setup.id, state="UNKNOWN", error="Create response omitted orderId; reconcile before entry")
                    return True
                await self._update(setup.id, state="ACCEPTED", order_id=str(order_id),
                    volume=float(result.get("vol", result.get("volume", 0))) or None)
                setup.position_size_usdt = float(result.get("notionalUsdt", notional))
                setup.entry = float(result.get("price", setup.entry))
                setup.stop_loss = float(result.get("stopLossPrice", setup.stop_loss))
                setup.take_profits = [float(result.get("takeProfitPrice", tp))]
                setup.risk_reward = float(result.get("riskReward", 2))
                from app.db.models import SetupRecord
                async with SessionLocal() as session:
                    saved = await session.get(SetupRecord, setup.id)
                    if saved:
                        saved.payload = setup.model_dump_json()
                        await session.commit()
                return True
            except MEXCAPIError as exc:
                uncertain = getattr(exc, "uncertain", False) or exc.code in {603, 2042}
                await self._update(setup.id, state="UNKNOWN" if uncertain else "REJECTED",
                    error="Ambiguous create; reconciliation required" if uncertain else f"MEXC rejected create (code {exc.code})")
                return uncertain
            except ValueError:
                await self._update(setup.id, state="UNKNOWN" if response_received else "REJECTED",
                    error="Accepted response could not be persisted" if response_received else "Contract or order validation failed")
                return response_received
            except Exception:
                # This includes a crash between exchange acceptance and DB commit.
                await self._update(setup.id, state="UNKNOWN", error="Create outcome unknown; reconciliation required")
                return True

    async def reconcile(self, setup: TradeSetup) -> tuple[str, float, float] | None:
        async with self.lock:
            async with SessionLocal() as session:
                record = await session.get(ExecutionRecord, setup.id)
            if record is None:
                return None
            if record.state == "CLOSED":
                return "CLOSED", record.actual_exit or setup.entry, record.realized_pnl or 0.0
            if record.state == "REJECTED":
                return "CANCELLED", setup.entry, 0.0
            if record.state not in RESERVED_STATES:
                return None
            try:
                order = await self.client.get_order_by_external_id(record.symbol, record.external_oid)
            except Exception:
                # Absence or network failure never proves an order was not accepted.
                return None
            order_id = str(order.get("orderId") or record.order_id or "")
            dealt = float(order.get("dealVol") or 0)
            state = int(order.get("state") or 0)
            if state in {4, 5} and dealt == 0:
                await self._update(setup.id, state="REJECTED", order_id=order_id or None)
                return "CANCELLED", setup.entry, 0.0
            expired = utcnow() - as_utc(record.created_at) > timedelta(minutes=self.settings.pending_order_max_age_minutes)
            terminal = state in {3, 4, 5}
            if not terminal and (dealt > 0 or expired):
                if record.cancel_state is None:
                    # Persist before the cancellation mutation. Never retry an ambiguous cancel blindly.
                    await self._update(setup.id, state="UNKNOWN", cancel_state="REQUESTED", error="Residual entry cancellation awaiting confirmation")
                    await self.client.cancel_order(record.symbol, order_id)
                # Keep monitoring the filled portion and its protection while the order settles.
            if terminal and record.cancel_state == "REQUESTED":
                await self._update(setup.id, cancel_state="CONFIRMED")
            if dealt == 0:
                return None
            position_id = str(order.get("positionId") or record.position_id or "")
            entry = float(order.get("dealAvgPrice") or record.actual_entry or setup.entry)
            if not position_id or position_id == "0":
                await self._update(setup.id, state="UNPROTECTED", error="Filled order has no position identity; protection cannot be verified")
                return None
            # Do not clear a persisted protection halt until all protection reads succeed.
            await self._update(setup.id, state="UNPROTECTED" if terminal else "UNKNOWN", order_id=order_id or None,
                position_id=position_id, actual_entry=entry, volume=dealt)
            positions = await self.client.get_positions(record.symbol)
            position = next((p for p in positions if str(p.get("positionId")) == position_id and float(p.get("holdVol", 0)) > 0), None)
            if position:
                stops = await self.client.get_stop_orders(record.symbol)
                def covers(s):
                    stop = float(s.get("stopLossPrice") or 0)
                    take = float(s.get("takeProfitPrice") or 0)
                    ordering = 0 < stop < entry < take if setup.direction == "LONG" else 0 < take < entry < stop
                    held = float(position.get("holdVol") or 0)
                    sl_vol = float(s.get("stopLossVol", s.get("vol", 0)) or 0)
                    tp_vol = float(s.get("takeProfitVol", s.get("vol", 0)) or 0)
                    return (str(s.get("positionId")) == position_id and int(s.get("state", 0)) == 1
                        and not s.get("isFinished", False) and ordering and min(sl_vol, tp_vol) >= held)
                protected = any(covers(s) for s in stops)
                if not protected:
                    await self._update(setup.id, state="UNPROTECTED", error="Exchange protection not confirmed; new entries halted")
                    logger.error("Position protection needs attention for {}", record.symbol)
                elif terminal:
                    await self._update(setup.id, state="FILLED", error=None)
                return None
            if not terminal:
                # A remaining entry may fill again; do not release its symbol/risk reservation.
                return None
            history = await self.client.get_history_positions(record.symbol)
            closed = next((p for p in history if str(p.get("positionId")) == position_id and int(p.get("state", 0)) == 3), None)
            if closed is None:
                return None
            # The exchange reports realised cash PnL; ticker touches never close a live trade.
            if "realised" not in closed and "closeProfitLoss" not in closed:
                return None
            pnl = float(closed.get("realised", closed.get("closeProfitLoss", 0)))
            exit_price = float(closed.get("closeAvgPrice") or entry)
            await self._update(setup.id, state="CLOSED", realized_pnl=pnl, actual_exit=exit_price, error=None)
            return "CLOSED", exit_price, pnl

    async def cancel_pending(self, setup_id: str) -> None:
        async with self.lock:
            async with SessionLocal() as session:
                record = await session.get(ExecutionRecord, setup_id)
            if record is None or record.state in {"REJECTED", "CLOSED"}:
                return
            if record.cancel_state == "REQUESTED":
                return
            order = await self.client.get_order_by_external_id(record.symbol, record.external_oid)
            if float(order.get("dealVol") or 0) > 0:
                raise ValueError("Filled position requires an explicit exchange close; cancellation is for unfilled entry only")
            await self._update(setup_id, state="UNKNOWN", cancel_state="REQUESTED", error="Manual cancellation awaiting exchange confirmation")
            await self.client.cancel_order(record.symbol, str(order.get("orderId") or record.order_id))
