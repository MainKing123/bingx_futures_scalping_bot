from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from app.backtest.schemas import BacktestRunRequest, BacktestRunStatusResponse, BacktestSummary
from app.db.engine import SessionLocal
from app.db.models import DailyStats, SetupRecord
from app.db.repository import aggregate_stats, record_to_setup, setups_query, update_setup_status
from app.schemas.setup import MarketOverview, TradeSetup
from app.tradingview import symbol_to_tv_candidates, timeframe_to_tv_interval

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


@router.get("/scanner/volatile-pairs", response_model=list[dict])
async def get_volatile_pairs(request: Request):
    return request.app.state.scanner.volatile_pairs


@router.post("/backtest/run", response_model=BacktestRunStatusResponse)
async def run_backtest(payload: BacktestRunRequest, request: Request):
    return await request.app.state.backtest_service.start(payload)


@router.get("/backtest/jobs/{job_id}", response_model=BacktestRunStatusResponse)
async def get_backtest_job_status(job_id: str, request: Request):
    status = request.app.state.backtest_service.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Backtest job not found")
    return status


@router.get("/backtest/jobs/{job_id}/result", response_model=BacktestSummary)
async def get_backtest_job_result(job_id: str, request: Request):
    status = request.app.state.backtest_service.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Backtest job not found")
    if status.status != "completed":
        raise HTTPException(status_code=409, detail=f"Backtest job status is {status.status}")
    result = request.app.state.backtest_service.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Backtest result not found")
    return result


@router.get("/backtest/latest", response_model=BacktestSummary)
async def get_latest_backtest(request: Request):
    result = request.app.state.backtest_service.get_latest_result()
    if result is None:
        raise HTTPException(status_code=404, detail="No completed backtest yet")
    return result


@router.get("/tradingview/symbol/{symbol}")
async def resolve_tradingview_symbol(symbol: str, request: Request):
    return {
        "input_symbol": symbol.upper(),
        "candidates": symbol_to_tv_candidates(symbol),
        "default_interval": timeframe_to_tv_interval(request.app.state.settings.ltf_timeframe),
    }


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
        "volatility_pool_size": settings.volatility_pool_size,
        "volatility_lookback_candles": settings.volatility_lookback_candles,
        "volatility_interval": settings.volatility_interval,
        "max_poi_distance_pct": settings.max_poi_distance_pct,
        "risk_per_trade_percent": settings.risk_per_trade_percent,
        "max_open_setups": settings.max_open_setups,
        "daily_loss_limit_percent": settings.daily_loss_limit_percent,
        "active_sessions": settings.active_sessions,
        "auto_execution": settings.auto_execution,
        "backtest_fee_bps": settings.backtest_fee_bps,
        "backtest_slippage_bps": settings.backtest_slippage_bps,
        "backtest_cooldown_candles": settings.backtest_cooldown_candles,
        "backtest_default_lookback_days": settings.backtest_default_lookback_days,
        "backtest_max_lookback_days": settings.backtest_max_lookback_days,
    }


@router.patch("/config")
async def update_config(updates: dict, request: Request):
    settings = request.app.state.settings
    updatable = {
        "top_pairs_count",
        "min_daily_volume_usd",
        "volatility_pool_size",
        "volatility_lookback_candles",
        "volatility_interval",
        "max_poi_distance_pct",
        "risk_per_trade_percent",
        "max_open_setups",
        "daily_loss_limit_percent",
        "active_sessions",
        "min_risk_reward",
        "backtest_fee_bps",
        "backtest_slippage_bps",
        "backtest_cooldown_candles",
        "backtest_default_lookback_days",
        "backtest_max_lookback_days",
    }
    for key, value in updates.items():
        if key in updatable and hasattr(settings, key):
            setattr(settings, key, value)
    return await get_config(request)
