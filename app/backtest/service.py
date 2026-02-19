from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from loguru import logger

from app.api.ws_manager import WSManager
from app.backtest.engine import BacktestEngine
from app.backtest.profiles import build_profile
from app.backtest.schemas import BacktestRunRequest, BacktestRunStatusResponse, BacktestSummary
from app.config import Settings
from app.exchange.client import BingXClient

INTERVAL_TO_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
}


@dataclass
class BacktestJob:
    job_id: str
    request: BacktestRunRequest
    lookback_days: int
    ltf_timeframe: str
    htf_timeframe: str
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    result: BacktestSummary | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BacktestService:
    def __init__(
        self,
        client: BingXClient,
        settings: Settings,
        scanner,
        ws_manager: WSManager | None = None,
        engine: BacktestEngine | None = None,
    ):
        self.client = client
        self.settings = settings
        self.scanner = scanner
        self.ws_manager = ws_manager
        self.engine = engine or BacktestEngine(client, settings)
        self.jobs: dict[str, BacktestJob] = {}
        self.latest_completed: BacktestSummary | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _interval_minutes(interval: str) -> int:
        return INTERVAL_TO_MINUTES.get(interval, 5)

    def _normalize_request(self, payload: BacktestRunRequest) -> tuple[int, str, str]:
        lookback = payload.lookback_days if payload.lookback_days is not None else self.settings.backtest_default_lookback_days
        lookback = max(1, min(int(lookback), self.settings.backtest_max_lookback_days))

        if payload.strategy == "crt_ict":
            allowed = list(self.settings.crt_entry_timeframes or ["5m", "15m"])
            ltf = (payload.ltf_timeframe or allowed[0]).lower()
            if ltf not in allowed:
                ltf = allowed[0]
            if payload.mode == "batch" and ltf not in {"5m", "15m"}:
                ltf = "5m"
            htf = "4h"
            return lookback, ltf, htf

        ltf = payload.ltf_timeframe or self.settings.ltf_timeframe
        htf = payload.htf_timeframe or self.settings.htf_timeframe
        if payload.mode == "batch" and self._interval_minutes(ltf) < 5:
            ltf = "5m"
        return lookback, ltf, htf

    async def _broadcast(self, event: str, data: dict):
        if self.ws_manager is None:
            return
        try:
            await self.ws_manager.broadcast(event, data)
        except Exception as exc:
            logger.warning(f"Backtest WS broadcast failed: {exc}")

    async def start(self, payload: BacktestRunRequest) -> BacktestRunStatusResponse:
        lookback, ltf, htf = self._normalize_request(payload)
        job_id = uuid4().hex
        job = BacktestJob(job_id=job_id, request=payload, lookback_days=lookback, ltf_timeframe=ltf, htf_timeframe=htf)
        async with self._lock:
            self.jobs[job_id] = job
        asyncio.create_task(self._run_job(job_id))
        await self._broadcast("backtest_job_update", self._status_payload(job))
        return self._to_status(job)

    async def _resolve_universe(self, job: BacktestJob) -> list[str]:
        req = job.request
        if req.mode == "single":
            return [str(req.symbol).upper()]

        pairs = getattr(self.scanner, "volatile_pairs", []) or []
        if pairs:
            symbols = [str(item.get("symbol", "")).upper() for item in pairs if isinstance(item, dict)]
            symbols = [s for s in symbols if s]
            if symbols:
                return symbols[: self.settings.top_pairs_count]

        ranked = await self.client.get_top_volatile_symbols(
            limit=self.settings.top_pairs_count,
            min_volume_usd=self.settings.min_daily_volume_usd,
            pool_size=self.settings.volatility_pool_size,
            interval=self.settings.volatility_interval,
            lookback=self.settings.volatility_lookback_candles,
        )
        return [str(item.get("symbol", "")).upper() for item in ranked if isinstance(item, dict) and item.get("symbol")]

    async def _run_job(self, job_id: str):
        job = self.jobs[job_id]
        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)
        await self._broadcast("backtest_job_update", self._status_payload(job))

        started_at = datetime.now(timezone.utc)
        try:
            universe = await self._resolve_universe(job)
            if not universe:
                raise RuntimeError("No symbols available for backtest universe")

            profile = build_profile(job.request.profile, self.settings)
            results = []
            for index, symbol in enumerate(universe, start=1):
                symbol_result = await self.engine.run_symbol(
                    symbol,
                    lookback_days=job.lookback_days,
                    ltf_timeframe=job.ltf_timeframe,
                    htf_timeframe=job.htf_timeframe,
                    profile=profile,
                    fee_bps=self.settings.backtest_fee_bps,
                    slippage_bps=self.settings.backtest_slippage_bps,
                    cooldown_candles=self.settings.backtest_cooldown_candles,
                    strategy=job.request.strategy,
                )
                results.append(symbol_result)
                job.progress = index / len(universe)
                job.updated_at = datetime.now(timezone.utc)
                await self._broadcast("backtest_job_update", self._status_payload(job))

            trades_count = sum(x.trades_count for x in results)
            wins = sum(x.wins for x in results)
            losses = sum(x.losses for x in results)
            gross_profit = sum(max(x.total_pnl_percent, 0.0) for x in results)
            gross_loss = sum(min(x.total_pnl_percent, 0.0) for x in results)
            total_pnl = sum(x.total_pnl_percent for x in results)
            expectancy = (total_pnl / trades_count) if trades_count else 0.0
            if gross_loss < 0:
                profit_factor = gross_profit / abs(gross_loss)
            else:
                profit_factor = gross_profit if gross_profit > 0 else 0.0

            summary = BacktestSummary(
                job_id=job.job_id,
                mode=job.request.mode,
                strategy=job.request.strategy,
                profile=profile.name,
                lookback_days=job.lookback_days,
                ltf_timeframe=job.ltf_timeframe,
                htf_timeframe=job.htf_timeframe,
                htf_context="1D+4H" if job.request.strategy == "crt_ict" else job.htf_timeframe,
                universe=universe,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                trades_count=trades_count,
                wins=wins,
                losses=losses,
                win_rate=round((wins / trades_count * 100), 4) if trades_count else 0.0,
                expectancy=round(expectancy, 6),
                profit_factor=round(profit_factor, 6),
                max_drawdown=round(max((x.max_drawdown for x in results), default=0.0), 6),
                total_pnl_percent=round(total_pnl, 6),
                symbol_results=results,
            )
            job.status = "completed"
            job.progress = 1.0
            job.result = summary
            job.updated_at = datetime.now(timezone.utc)
            self.latest_completed = summary
            await self._broadcast("backtest_completed", {"job_id": job.job_id, "trades_count": trades_count})
            await self._broadcast("backtest_job_update", self._status_payload(job))
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = datetime.now(timezone.utc)
            logger.exception(f"Backtest job {job.job_id} failed: {exc}")
            await self._broadcast("backtest_job_update", self._status_payload(job))

    def _status_payload(self, job: BacktestJob) -> dict:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "mode": job.request.mode,
            "strategy": job.request.strategy,
            "profile": job.request.profile,
            "progress": round(job.progress, 6),
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "error": job.error,
        }

    def _to_status(self, job: BacktestJob) -> BacktestRunStatusResponse:
        return BacktestRunStatusResponse(
            job_id=job.job_id,
            status=job.status,  # type: ignore[arg-type]
            mode=job.request.mode,
            strategy=job.request.strategy,
            profile=job.request.profile,
            progress=round(job.progress, 6),
            created_at=job.created_at,
            updated_at=job.updated_at,
            error=job.error,
        )

    def get_status(self, job_id: str) -> BacktestRunStatusResponse | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        return self._to_status(job)

    def get_result(self, job_id: str) -> BacktestSummary | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        return job.result

    def get_latest_result(self) -> BacktestSummary | None:
        return self.latest_completed
