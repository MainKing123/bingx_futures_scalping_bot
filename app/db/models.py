from __future__ import annotations
from datetime import datetime
from sqlalchemy import Float, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SetupRecord(Base):
    __tablename__ = "volium_setups"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime]
    symbol: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, index=True)
    execution_mode: Mapped[str] = mapped_column(String, default="paper")
    payload: Mapped[str]
    closed_at: Mapped[datetime | None]
    pnl_usdt: Mapped[float | None] = mapped_column(Float)
    paper_filled_at: Mapped[datetime | None]
    paper_last_bar_at: Mapped[datetime | None]


class ExecutionRecord(Base):
    __tablename__ = "executions"
    setup_id: Mapped[str] = mapped_column(String, primary_key=True)
    external_oid: Mapped[str] = mapped_column(String, unique=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    direction: Mapped[str]
    state: Mapped[str] = mapped_column(String, index=True)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    order_id: Mapped[str | None]
    position_id: Mapped[str | None]
    volume: Mapped[float | None] = mapped_column(Float)
    actual_entry: Mapped[float | None] = mapped_column(Float)
    realized_pnl: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None]
    cancel_state: Mapped[str | None]
    actual_exit: Mapped[float | None] = mapped_column(Float)
