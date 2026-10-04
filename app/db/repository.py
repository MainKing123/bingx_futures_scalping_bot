from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import select
from app.db.models import SetupRecord
from app.schemas.setup import TradeSetup


def record_to_setup(record):
    setup = TradeSetup.model_validate_json(record.payload)
    setup.status = record.status
    return setup


async def save_setup(session, setup, execution_mode="paper"):
    session.add(SetupRecord(id=setup.id, created_at=setup.timestamp, symbol=setup.symbol,
        status=setup.status, execution_mode=execution_mode, payload=setup.model_dump_json()))
    await session.commit()


async def update_setup_status(session, setup_id, status, pnl_usdt=None):
    rec = await session.get(SetupRecord, setup_id)
    if rec:
        rec.status = status
        rec.pnl_usdt = pnl_usdt
        rec.closed_at = datetime.now(timezone.utc)
        await session.commit()


def setups_query(status=None):
    query = select(SetupRecord).order_by(SetupRecord.created_at.desc())
    return query.where(SetupRecord.status == status) if status else query
