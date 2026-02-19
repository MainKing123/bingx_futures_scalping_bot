from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from app.db.engine import SessionLocal
from app.db.models import DailyStats, SetupRecord
from app.db.repository import aggregate_stats, record_to_setup, setups_query, update_setup_status
from app.schemas.setup import MarketOverview, TradeSetup

router = APIRouter(prefix="/api")


@router.get("/setups", response_model=list[TradeSetup])
async def get_setups(status: str | None = None, direction: str | None = None, confidence: str | None = None, symbol: str | None = None, limit: int = 50):
    async with SessionLocal() as session:
        rows = await session.scalars(setups_query(status, direction, confidence, symbol).limit(limit))
        return [record_to_setup(row) for row in rows.all()]


@router.get("/setups/{setup_id}", response_model=TradeSetup)
async def get_setup_detail(setup_id: str):
    async with SessionLocal() as session:
        rec = await session.get(SetupRecord, setup_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Setup not found")
        return record_to_setup(rec)


@router.get("/watchlist", response_model=list[dict])
async def get_watchlist(request: Request):
    scanner = request.app.state.scanner
    return [
        {
            "symbol": x.symbol,
            "poi_zone": x.poi_zone,
            "added_at": x.added_at,
            "last_checked": x.last_checked,
            "bias": x.htf_analysis.bias,
        }
        for x in scanner.watchlist.values()
    ]


@router.get("/market-structure/{symbol}", response_model=MarketOverview)
async def get_market_structure(symbol: str, request: Request):
    return await request.app.state.engine.get_market_overview(symbol)


@router.get("/stats")
async def get_stats(period: str = "7d"):
    async with SessionLocal() as session:
        stats = await aggregate_stats(session)
        stats["period"] = period
        return stats


@router.get("/stats/equity-curve", response_model=list[dict])
async def get_equity_curve(period: str = "30d"):
    async with SessionLocal() as session:
        rows = await session.scalars(select(DailyStats).order_by(DailyStats.date.asc()))
        cumulative = 0.0
        curve = []
        for row in rows.all():
            cumulative += row.total_pnl_percent
            curve.append({"date": row.date, "cumulative_pnl_percent": round(cumulative, 4)})
        return curve


@router.get("/account/balance")
async def get_account_balance(request: Request):
    return await request.app.state.client.get_balance()


@router.get("/account/positions")
async def get_account_positions(request: Request):
    return await request.app.state.client.get_positions()


@router.patch("/setups/{setup_id}/cancel")
async def cancel_setup(setup_id: str):
    async with SessionLocal() as session:
        await update_setup_status(session, setup_id, "CANCELLED")
        rec = await session.get(SetupRecord, setup_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Setup not found")
        return record_to_setup(rec)


@router.get("/config")
async def get_config(request: Request):
    settings = request.app.state.settings
    return {
        "top_pairs_count": settings.top_pairs_count,
        "scan_interval_seconds": settings.scan_interval_seconds,
        "min_daily_volume_usd": settings.min_daily_volume_usd,
        "risk_per_trade_percent": settings.risk_per_trade_percent,
        "max_open_setups": settings.max_open_setups,
        "daily_loss_limit_percent": settings.daily_loss_limit_percent,
        "active_sessions": settings.active_sessions,
        "auto_execution": settings.auto_execution,
    }


@router.patch("/config")
async def update_config(updates: dict, request: Request):
    settings = request.app.state.settings
    updatable = {
        "risk_per_trade_percent",
        "max_open_setups",
        "daily_loss_limit_percent",
        "active_sessions",
        "min_risk_reward",
    }
    for key, value in updates.items():
        if key in updatable and hasattr(settings, key):
            setattr(settings, key, value)
    return await get_config(request)
