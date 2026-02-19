from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SetupRecord(Base):
    __tablename__ = "setups"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime]
    symbol: Mapped[str] = mapped_column(String, index=True)
    direction: Mapped[str]
    setup_type: Mapped[str]
    htf_bias: Mapped[str]
    entry: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profits: Mapped[str]
    risk_reward: Mapped[float] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String, index=True)
    confluences: Mapped[str]
    status: Mapped[str] = mapped_column(String, index=True)
    closed_at: Mapped[datetime | None]
    pnl_percent: Mapped[float | None] = mapped_column(Float)
    position_size: Mapped[float | None] = mapped_column(Float)
    chart_data: Mapped[str | None]
    entry_order_id: Mapped[str | None]
    sl_order_id: Mapped[str | None]
    tp_order_id: Mapped[str | None]


class DailyStats(Base):
    __tablename__ = "daily_stats"

    date: Mapped[str] = mapped_column(String, primary_key=True)
    total_setups: Mapped[int] = mapped_column(default=0)
    wins: Mapped[int] = mapped_column(default=0)
    losses: Mapped[int] = mapped_column(default=0)
    total_pnl_percent: Mapped[float] = mapped_column(default=0.0)
    best_rr: Mapped[float | None]
    worst_rr: Mapped[float | None]
