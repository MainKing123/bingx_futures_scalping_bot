from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DailyStats, SetupRecord
from app.schemas.setup import TradeSetup


def setup_to_record(setup: TradeSetup) -> SetupRecord:
    return SetupRecord(
        id=setup.id,
        created_at=setup.timestamp,
        symbol=setup.symbol,
        direction=setup.direction,
        setup_type=setup.setup_type,
        htf_bias=setup.htf_bias,
        entry=setup.entry,
        stop_loss=setup.stop_loss,
        take_profits=json.dumps(setup.take_profits),
        risk_reward=setup.risk_reward,
        confidence=setup.confidence,
        confluences=json.dumps(setup.confluences),
        status=setup.status,
        closed_at=None,
        pnl_percent=None,
        position_size=setup.position_size_usdt,
        chart_data=setup.chart_data.model_dump_json() if setup.chart_data else None,
        entry_order_id=None,
        sl_order_id=None,
        tp_order_id=None,
    )


def record_to_setup(record: SetupRecord) -> TradeSetup:
    return TradeSetup(
        id=record.id,
        timestamp=record.created_at,
        symbol=record.symbol,
        direction=record.direction,
        setup_type=record.setup_type,
        htf_bias=record.htf_bias,
        entry=record.entry,
        stop_loss=record.stop_loss,
        take_profits=json.loads(record.take_profits),
        risk_reward=record.risk_reward,
        confidence=record.confidence,
        confluences=json.loads(record.confluences),
        status=record.status,
        position_size_usdt=record.position_size,
        chart_data=None,
    )


async def save_setup(session: AsyncSession, setup: TradeSetup) -> None:
    session.add(setup_to_record(setup))
    await session.commit()


async def update_setup_status(session: AsyncSession, setup_id: str, status: str, pnl_percent: float | None = None) -> None:
    rec = await session.get(SetupRecord, setup_id)
    if rec is None:
        return
    rec.status = status
    rec.pnl_percent = pnl_percent
    rec.closed_at = datetime.now(timezone.utc)
    await session.commit()


async def increment_daily_stats(session: AsyncSession, setup: TradeSetup, pnl_percent: float | None = None) -> None:
    date_key = datetime.now(timezone.utc).date().isoformat()
    stats = await session.get(DailyStats, date_key)
    if stats is None:
        stats = DailyStats(date=date_key, total_setups=0, wins=0, losses=0, total_pnl_percent=0.0, best_rr=None, worst_rr=None)
        session.add(stats)
    stats.total_setups += 1
    if pnl_percent is not None:
        stats.total_pnl_percent += pnl_percent
        if pnl_percent >= 0:
            stats.wins += 1
        else:
            stats.losses += 1
    stats.best_rr = setup.risk_reward if stats.best_rr is None else max(stats.best_rr, setup.risk_reward)
    stats.worst_rr = setup.risk_reward if stats.worst_rr is None else min(stats.worst_rr, setup.risk_reward)
    await session.commit()


def setups_query(status: str | None = None, direction: str | None = None, confidence: str | None = None, symbol: str | None = None) -> Select[tuple[SetupRecord]]:
    query: Select[tuple[SetupRecord]] = select(SetupRecord).order_by(SetupRecord.created_at.desc())
    if status:
        query = query.where(SetupRecord.status == status)
    if direction:
        query = query.where(SetupRecord.direction == direction)
    if confidence:
        query = query.where(SetupRecord.confidence == confidence)
    if symbol:
        query = query.where(SetupRecord.symbol == symbol)
    return query


async def aggregate_stats(session: AsyncSession) -> dict:
    total = await session.scalar(select(func.count()).select_from(SetupRecord)) or 0
    wins = await session.scalar(select(func.count()).select_from(SetupRecord).where(SetupRecord.pnl_percent != None, SetupRecord.pnl_percent > 0)) or 0
    losses = await session.scalar(select(func.count()).select_from(SetupRecord).where(SetupRecord.pnl_percent != None, SetupRecord.pnl_percent < 0)) or 0
    avg_rr = await session.scalar(select(func.avg(SetupRecord.risk_reward))) or 0.0
    gross_profit = await session.scalar(select(func.coalesce(func.sum(SetupRecord.pnl_percent), 0.0)).where(SetupRecord.pnl_percent != None, SetupRecord.pnl_percent > 0)) or 0.0
    gross_loss = await session.scalar(select(func.coalesce(func.sum(SetupRecord.pnl_percent), 0.0)).where(SetupRecord.pnl_percent != None, SetupRecord.pnl_percent < 0)) or 0.0

    by_confidence = await session.execute(select(SetupRecord.confidence, func.count()).group_by(SetupRecord.confidence))
    by_type = await session.execute(select(SetupRecord.setup_type, func.count()).group_by(SetupRecord.setup_type))
    pnl_rows = await session.scalars(select(SetupRecord.pnl_percent).where(SetupRecord.pnl_percent != None).order_by(SetupRecord.closed_at.asc()))
    streak = 0
    max_streak = 0
    for pnl in pnl_rows.all():
        if pnl is not None and pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "total_setups": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round((wins / total) * 100, 2) if total else 0.0,
        "profit_factor": round(gross_profit / abs(gross_loss), 3) if gross_loss else 0.0,
        "avg_rr": round(float(avg_rr), 3),
        "max_consecutive_losses": max_streak,
        "breakdown_confidence": {k: v for k, v in by_confidence.all()},
        "breakdown_setup_type": {k: v for k, v in by_type.all()},
    }
