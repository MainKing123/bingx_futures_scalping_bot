from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select
from app.db.engine import SessionLocal
from app.db.models import ExecutionRecord, SetupRecord
from app.db.repository import record_to_setup, setups_query

router = APIRouter(prefix="/api")


@router.get("/status")
async def status(request: Request):
    state = request.app.state
    try:
        marked = state.risk_manager.balance if state.settings.auto_execution else getattr(state.tracker, "marked_equity", state.risk_manager.balance)
        if not isinstance(marked, (int, float)) or not math.isfinite(marked):
            marked = None
    except ValueError:
        marked = None
    return {"exchange":"MEXC", "mode":"live" if state.settings.auto_execution else "paper",
        "strategy":state.settings.volium_mode, "symbols":state.scanner.active_symbols,
        "strategy_profile":getattr(state.settings, "volium_strategy_profile", "v1_guarded"),
        "strategy_is_experimental":getattr(state.settings, "volium_strategy_profile", "").startswith("v5"),
        "universe_core_symbols":getattr(state.settings, "universe_core_symbols", []),
        "entries_paused":not state.risk_manager.can_open_setup() or bool(state.scanner.selection_error) or marked is None, "reserved_slots":state.risk_manager.open_setups,
        "equity_usdt":state.risk_manager.balance,
        "marked_equity_usdt":marked,
        "unrealized_pnl_usdt":None if marked is None else marked-state.risk_manager.balance,
        "daily_opening_equity_usdt":state.risk_manager.daily_opening_balance,
        "daily_opening_equity_source":state.risk_manager.daily_opening_balance_source,
        "daily_pnl_usdt":state.risk_manager.get_daily_pnl(),
        "last_scan_at":state.scanner.last_scan_at,"errors":state.scanner.last_errors,
        "execution_rejections":state.scanner.last_rejections,
        "leverage_cap":state.settings.default_leverage,
        "leverage_asset_caps":getattr(state.settings, "leverage_asset_caps", {}),
        "execution_cost_guard":getattr(state.settings, "execution_cost_guard_enabled", False),
        "session_utc3":state.settings.volium_sessions_utc3,
        "session_clock":state.settings.volium_session_clock,
        "market_sessions":state.settings.volium_market_sessions,
        "pair_selection":state.settings.pair_selection,
        "selected_pairs":state.scanner.selected_pairs,
        "selection_error":state.scanner.selection_error,
        "session_note":"BTC/ETH: 10:00–12:00 UTC+3; afternoon 16:30 needs an explicit end time",
        "risk_per_trade_percent":state.settings.risk_per_trade_percent}


@router.get("/config")
async def config(request: Request):
    # Credentials are excluded by their schema and never appear in this endpoint.
    return request.app.state.settings.model_dump(mode="json")


@router.get("/setups")
async def setups(limit: int = 100):
    async with SessionLocal() as session:
        rows = (await session.scalars(setups_query().limit(max(1,min(limit,500))))).all()
        return [{**record_to_setup(row).model_dump(mode="json"), "execution_mode":row.execution_mode,
            "paper_filled_at":row.paper_filled_at,"pnl_usdt":row.pnl_usdt} for row in rows]


@router.get("/executions")
async def executions():
    async with SessionLocal() as session:
        rows = (await session.scalars(select(ExecutionRecord).order_by(ExecutionRecord.created_at.desc()).limit(100))).all()
        return [{"setup_id":r.setup_id,"symbol":r.symbol,"state":r.state,"order_id":r.order_id,
                 "volume":r.volume,"actual_entry":r.actual_entry,"realized_pnl":r.realized_pnl,
                 "error":r.error,"updated_at":r.updated_at} for r in rows]


@router.post("/setups/{setup_id}/cancel")
async def cancel(setup_id: str, request: Request):
    try:
        return await request.app.state.tracker.cancel(setup_id)
    except ValueError as exc:
        raise HTTPException(409,detail=str(exc)) from None
    except Exception:
        raise HTTPException(503,detail="Exchange confirmation unavailable; state retained") from None
