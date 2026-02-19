from __future__ import annotations

from fastapi import APIRouter

from app.schemas.setup import MarketOverview, TradeSetup

router = APIRouter(prefix="/api")

SETUPS: dict[str, TradeSetup] = {}
WATCHLIST: dict[str, dict] = {}


@router.get("/setups", response_model=list[TradeSetup])
async def get_setups(status: str | None = None, direction: str | None = None, confidence: str | None = None, symbol: str | None = None, limit: int = 50):
    items = list(SETUPS.values())
    if status:
        items = [i for i in items if i.status == status]
    if direction:
        items = [i for i in items if i.direction == direction]
    if confidence:
        items = [i for i in items if i.confidence == confidence]
    if symbol:
        items = [i for i in items if i.symbol == symbol]
    return items[:limit]


@router.get("/setups/{setup_id}", response_model=TradeSetup)
async def get_setup_detail(setup_id: str):
    return SETUPS[setup_id]


@router.get("/watchlist", response_model=list[dict])
async def get_watchlist():
    return list(WATCHLIST.values())


@router.get("/market-structure/{symbol}", response_model=MarketOverview)
async def get_market_structure(symbol: str):
    return MarketOverview(symbol=symbol, trend="RANGING", bias="RANGING", active_obs=[], active_fvgs=[])


@router.get("/stats")
async def get_stats(period: str = "7d"):
    return {"period": period, "total_setups": len(SETUPS)}


@router.get("/stats/equity-curve", response_model=list[dict])
async def get_equity_curve(period: str = "30d"):
    return []


@router.get("/account/balance")
async def get_account_balance():
    return {"balance": 0}


@router.get("/account/positions")
async def get_account_positions():
    return []


@router.patch("/setups/{setup_id}/cancel")
async def cancel_setup(setup_id: str):
    setup = SETUPS[setup_id]
    setup.status = "CANCELLED"
    return setup


@router.get("/config")
async def get_config():
    return {"ok": True}


@router.patch("/config")
async def update_config(updates: dict):
    return updates
