"""Frozen public-data experiments for the video strategy, without private API calls."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.config import Settings
from app.exchange.client import MEXCClient
from app.replay import SECONDS, replay, required_timeframes
from app.strategy.volium import analyze_volium_from_df

VIDEO_URL = "https://www.youtube.com/watch?v=jYvt1hSTPxc"
DEFAULT_OUTPUTS = Path("outputs")
SYMBOLS = ("ZEC_USDT", "SOL_USDT", "DOGE_USDT", "XRP_USDT", "ETH_USDT")
MAX_DAYS = {"intraday": 180, "scalp": 30, "swing": 365, "swing_weekly": 365}
BASELINE_SESSIONS = [("10:00", "12:00"), ("16:30", "18:00")]
MARKET_SESSIONS = [("Europe/London", "08:00", "10:00"), ("America/New_York", "09:30", "11:00")]
CACHE_LIMITS = {"1m": 30 * 1440 + 130, "5m": 180 * 288 + 130,
                "1h": 365 * 24 + 130, "4h": 365 * 6 + 130, "1d": 600, "1w": 182}


@dataclass(frozen=True)
class Scenario:
    name: str
    mode: str
    days: int
    end_offset_days: int = 0
    swing_context: str = "1d"
    fee_bps: float = 5
    slippage_bps: float = 2
    sessions: str = "both"
    session_clock: str = "fixed_utc3"
    body_ratio: float = 0.6
    pivot_lookback: int = 2
    require_origin_sweep: bool = True
    execution: str = "strategy"
    category: str = "baseline"

    def settings(self):
        return Settings(_env_file=None, mexc_api_key="", mexc_api_secret="", auto_execution=False,
            volium_mode=self.mode, volium_swing_context=self.swing_context,
            risk_per_trade_percent=0.5, account_balance_usdt=1000,
            default_leverage=3, daily_loss_limit_percent=2, pending_order_max_age_minutes=30,
            paper_fee_bps=self.fee_bps, paper_slippage_bps=self.slippage_bps,
            volium_sessions_utc3=BASELINE_SESSIONS if self.sessions == "both" else BASELINE_SESSIONS[:1],
            volium_session_clock=self.session_clock,
            volium_market_sessions=MARKET_SESSIONS if self.sessions == "both" else MARKET_SESSIONS[:1],
            volium_min_body_ratio=self.body_ratio, volium_swing_lookback=self.pivot_lookback,
            volium_require_daily_origin_sweep=self.require_origin_sweep)


def build_scenarios(include_weekly=True, include_fine=True):
    scenarios = []
    specs = [("intraday", "intraday", "1d", 180, [30, 60, 90], 60),
             ("scalp", "scalp", "1d", 30, [7, 14], 10),
             ("swing", "swing", "1d", 365, [90, 180], 120)]
    if include_weekly:
        specs.append(("swing_weekly", "swing", "1w", 365, [90, 180], 120))
    for label, mode, context, full_days, overlaps, segment in specs:
        common = dict(mode=mode, swing_context=context)
        scenarios.append(Scenario(f"{label}_baseline_{full_days}d", days=full_days, **common))
        for days in overlaps:
            scenarios.append(Scenario(f"{label}_overlap_{days}d", days=days, category="overlap", **common))
        scenarios.append(Scenario(f"{label}_stress_{full_days}d", days=full_days,
            fee_bps=10, slippage_bps=5, category="cost_stress", **common))
        for index in range(3):
            scenarios.append(Scenario(f"{label}_segment_{index+1}", days=segment,
                end_offset_days=index*segment, category="disjoint_segment", **common))
        if mode != "swing":
            scenarios.append(Scenario(f"{label}_morning_only", days=full_days,
                sessions="morning", category="sensitivity", **common))
            scenarios.append(Scenario(f"{label}_HYPOTHESIS_market_local", days=full_days,
                session_clock="market_local", category="session_clock_hypothesis", **common))
            scenarios.append(Scenario(f"{label}_HYPOTHESIS_market_local_morning", days=full_days,
                session_clock="market_local", sessions="morning", category="session_clock_hypothesis", **common))
        for body in (0.5, 0.7):
            scenarios.append(Scenario(f"{label}_body_{body}", days=full_days,
                body_ratio=body, category="sensitivity", **common))
        scenarios.append(Scenario(f"{label}_pivot_3", days=full_days,
            pivot_lookback=3, category="sensitivity", **common))
        if include_fine and mode != "scalp":
            scenarios.append(Scenario(f"{label}_fine_execution_30d", days=30,
                execution="1m", category="execution_resolution", **common))
    scenarios.append(Scenario("intraday_DEVIATION_origin_sweep_off", mode="intraday", days=180,
        require_origin_sweep=False, category="ablation_deviation"))
    return scenarios


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_metadata(frame, timeframe):
    seconds = SECONDS[timeframe]
    gaps = frame.index.to_series().diff().dropna().dt.total_seconds()
    missing = int(sum(max(0, round(value / seconds) - 1) for value in gaps))
    return {"bars": len(frame), "first_open_utc": frame.index[0].isoformat(),
            "last_open_utc": frame.index[-1].isoformat(),
            "last_close_utc": (frame.index[-1] + pd.Timedelta(seconds=seconds)).isoformat(),
            "missing_intervals": missing}


def load_frame(path):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    if frame.empty or frame.index.has_duplicates:
        raise RuntimeError(f"Invalid empty/duplicate cache: {path.name}")
    required = ["open", "high", "low", "close"]
    if not set(required).issubset(frame) or frame[required].isna().any().any():
        raise RuntimeError(f"Invalid OHLC cache: {path.name}")
    return frame


async def fetch_snapshot(cache, symbols, refresh=False, cache_only=False):
    manifest_path = cache / "snapshot.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() and not refresh else None
    if manifest is None and cache_only:
        raise RuntimeError("--cache-only requires an existing snapshot")
    cache.mkdir(parents=True, exist_ok=True)
    settings = Settings(_env_file=None, mexc_api_key="", mexc_api_secret="", auto_execution=False)
    client = MEXCClient(settings)
    frames = {}
    try:
        if manifest is None:
            server_ms = await client.get_server_time(refresh=True)
            snapshot = datetime.fromtimestamp(server_ms / 1000, timezone.utc)
            manifest = {"source": "MEXC public futures OHLC", "server_snapshot_utc": snapshot.isoformat(),
                        "snapshot_ms": server_ms, "symbols": [], "frames": {}}
        else:
            server_ms = manifest["snapshot_ms"]
        print(f"Frozen public server snapshot: {manifest['server_snapshot_utc']}", flush=True)
        for symbol in symbols:
            frames[symbol] = {}
            metadata = manifest["frames"].setdefault(symbol, {})
            for tf, limit in CACHE_LIMITS.items():
                path = cache / f"{symbol}_{tf}.csv"
                recorded = metadata.get(tf)
                if recorded:
                    if not path.exists() or sha256_file(path) != recorded["sha256"]:
                        raise RuntimeError(f"Cache missing or modified: {path.name}; use --refresh-cache")
                    loaded = load_frame(path)
                    bounded = loaded.loc[loaded.index + pd.Timedelta(seconds=SECONDS[tf]) <= pd.Timestamp(manifest["server_snapshot_utc"])]
                    if len(bounded) != len(loaded):
                        bounded.to_csv(path, index_label="timestamp")
                        metadata[tf] = {**frame_metadata(bounded, tf), "requested_bars": limit, "sha256": sha256_file(path)}
                        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    frames[symbol][tf] = bounded
                    continue
                if cache_only:
                    raise RuntimeError(f"Cache missing for {symbol} {tf}; run --fetch-only first")
                print(f"Fetching {symbol} {tf}: up to {limit:,} closed public candles", flush=True)
                frame = await client.get_klines(symbol, tf, limit=limit, end_time=server_ms)
                frame = frame.loc[frame.index + pd.Timedelta(seconds=SECONDS[tf]) <= pd.Timestamp(manifest["server_snapshot_utc"])]
                if frame.empty:
                    raise RuntimeError(f"No public candles for {symbol} {tf}")
                frame.to_csv(path, index_label="timestamp")
                frames[symbol][tf] = frame
                metadata[tf] = {**frame_metadata(frame, tf), "requested_bars": limit, "sha256": sha256_file(path)}
                manifest["symbols"] = list(dict.fromkeys([*manifest["symbols"], symbol]))
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(f"Cached {symbol} {tf}: {len(frame):,} bars, "
                      f"{frame.index[0].isoformat()} to {frame.index[-1].isoformat()}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return frames, manifest
    finally:
        await client.close()


class SignalCache:
    def __init__(self):
        self.data = {}
        self.broad = {}
        self.hits = self.misses = 0
        self.broad_rejections = self.broad_evaluations = 0

    def metrics(self):
        return {"hits": self.hits, "misses": self.misses,
                "broad_rejections": self.broad_rejections, "broad_evaluations": self.broad_evaluations}

    def provider(self, scenario):
        settings = scenario.settings()
        strategy = {key: value for key, value in settings.model_dump().items() if key.startswith("volium_")}
        config_hash = hashlib.sha256(json.dumps(strategy, sort_keys=True).encode()).hexdigest()
        broad_settings = settings.model_copy(update={"volium_session_enabled": False, "volium_min_body_ratio": 0.5})
        structural = {key: value for key, value in strategy.items() if key not in {
            "volium_sessions_utc3", "volium_session_clock", "volium_market_sessions", "volium_session_enabled"}}
        structural["volium_min_body_ratio"] = 0.5
        broad_hash = hashlib.sha256(json.dumps(structural, sort_keys=True).encode()).hexdigest()

        def analyze(**kwargs):
            key = (kwargs["symbol"], config_hash, pd.Timestamp(kwargs["now"]).value)
            if key in self.data:
                self.hits += 1
                found = self.data[key]
            else:
                self.misses += 1
                broad_key = (kwargs["symbol"], broad_hash, pd.Timestamp(kwargs["now"]).value)
                found = None
                # Higher body thresholds and enabled session filters only remove
                # candidate sweeps. A broad rejection is shared; broad candidates
                # always undergo exact analysis, since sessions may select another sweep.
                if settings.volium_min_body_ratio >= 0.5:
                    if broad_key not in self.broad:
                        self.broad_evaluations += 1
                        self.broad[broad_key] = analyze_volium_from_df(**{**kwargs, "settings": broad_settings})
                    if self.broad[broad_key] is None:
                        self.broad_rejections += 1
                    else:
                        found = analyze_volium_from_df(**kwargs)
                else:
                    found = analyze_volium_from_df(**kwargs)
                self.data[key] = found
            return found.model_copy(deep=True) if found is not None else None
        return analyze


async def funding_snapshot(cache, manifest, symbols, cache_only=False):
    """Freeze actual public settlement rates to the same end timestamp as OHLC."""
    rates_by_symbol = {}
    metadata = manifest.setdefault("funding", {})
    missing = []
    for symbol in symbols:
        path = cache / f"{symbol}_funding.csv"
        info = metadata.get(symbol)
        if info and info.get("status") == "available":
            if not path.exists() or sha256_file(path) != info["sha256"]:
                raise RuntimeError(f"Funding cache modified or missing: {path.name}")
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            frame.index = pd.to_datetime(frame.index, utc=True)
            rates_by_symbol[symbol] = frame["funding_rate"].sort_index()
        elif info:
            rates_by_symbol[symbol] = None
        else:
            missing.append(symbol)
    if missing and cache_only:
        raise RuntimeError("Funding snapshot missing; run --fetch-only first")
    if missing:
        settings = Settings(_env_file=None, mexc_api_key="", mexc_api_secret="", auto_execution=False)
        client = MEXCClient(settings)
        end = pd.Timestamp(manifest["server_snapshot_utc"])
        since = end - pd.Timedelta(days=400)
        try:
            for symbol in missing:
                print(f"Fetching actual public funding settlements: {symbol}, 400 days", flush=True)
                try:
                    rates = await client.get_funding_history(symbol, int(since.timestamp() * 1000))
                    rates = rates.loc[(rates.index >= since) & (rates.index <= end)]
                    if rates.empty:
                        raise RuntimeError("No funding settlements returned")
                    path = cache / f"{symbol}_funding.csv"
                    rates.to_csv(path, index_label="settlement_utc")
                    metadata[symbol] = {"status": "available", "settlements": len(rates),
                        "requested_start_utc": since.isoformat(), "first_settlement_utc": rates.index[0].isoformat(),
                        "last_settlement_utc": rates.index[-1].isoformat(),
                        "maximum_gap_hours": rates.index.to_series().diff().dt.total_seconds().max() / 3600,
                        "sha256": sha256_file(path)}
                    rates_by_symbol[symbol] = rates
                    print(f"Cached actual funding {symbol}: {len(rates)} settlements", flush=True)
                except Exception as exc:
                    metadata[symbol] = {"status": "unavailable", "reason": str(exc)}
                    rates_by_symbol[symbol] = None
                    print(f"Funding unavailable for {symbol}: {type(exc).__name__}", flush=True)
        finally:
            await client.close()
        (cache / "snapshot.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rates_by_symbol


def run_case(symbol, full_frames, scenario, snapshot, cache, funding_rates=None):
    settings = scenario.settings()
    end_at = snapshot - timedelta(days=scenario.end_offset_days)
    start_at = end_at - timedelta(days=scenario.days)
    tfs = required_timeframes(settings)
    frames = {tf: full_frames[tf].loc[
        full_frames[tf].index + pd.Timedelta(seconds=SECONDS[tf]) <= pd.Timestamp(end_at)]
        for tf in tfs}
    entry_tf = tfs[-1]
    entry = frames[entry_tf]
    observed = entry.loc[(entry.index >= pd.Timestamp(start_at)) &
                         (entry.index + pd.Timedelta(seconds=SECONDS[entry_tf]) <= pd.Timestamp(end_at))]
    kwargs = {"start_at": start_at, "signal_provider": cache.provider(scenario)}
    if "funding_rates" not in inspect.signature(replay).parameters:
        raise RuntimeError("Replay funding_rates support is required before running the suite")
    kwargs["funding_rates"] = funding_rates
    if scenario.execution == "1m":
        if "execution_frame" not in inspect.signature(replay).parameters:
            raise RuntimeError("Replay execution_frame support is required for finer execution cases")
        finer = full_frames["1m"].loc[
            (full_frames["1m"].index >= pd.Timestamp(start_at)) &
            (full_frames["1m"].index + pd.Timedelta(minutes=1) <= pd.Timestamp(end_at))]
        kwargs["execution_frame"] = finer
    started = time.monotonic()
    result = replay(symbol, frames, settings, **kwargs)
    trades = result.get("trades", [])
    filled_open = int(bool(result.get("unfinished_position")))
    unfilled = result.get("unfilled_signals", max(0, result["signals"] - len(trades) - filled_open))
    expected = max(0, int(scenario.days * 86400 / SECONDS[entry_tf]) - 1)
    result.update({
        "scenario": asdict(scenario), "requested_start_utc": start_at.isoformat(),
        "requested_end_utc": end_at.isoformat(), "entry_timeframe": entry_tf,
        "observed_entry_bars": len(observed), "expected_entry_bars_approx": expected,
        "entry_coverage_percent": min(100, len(observed) / expected * 100) if expected else None,
        "actual_first_entry_open_utc": observed.index[0].isoformat() if len(observed) else None,
        "actual_last_entry_close_utc": (observed.index[-1] + pd.Timedelta(seconds=SECONDS[entry_tf])).isoformat() if len(observed) else None,
        "long_trades": sum(t["direction"] == "LONG" for t in trades),
        "short_trades": sum(t["direction"] == "SHORT" for t in trades),
        "long_pnl_usdt": sum(t["pnl_usdt"] for t in trades if t["direction"] == "LONG"),
        "short_pnl_usdt": sum(t["pnl_usdt"] for t in trades if t["direction"] == "SHORT"),
        "unfilled_signals": unfilled, "elapsed_seconds": round(time.monotonic() - started, 3),
        "funding_data_available": funding_rates is not None,
    })
    return result


def summary_rows(results):
    rows = []
    for result in results:
        scenario = result["scenario"]
        rows.append({
            "scenario": scenario["name"], "category": scenario["category"],
            "symbol": result["symbol"], "mode": result["mode"], "swing_context": scenario["swing_context"],
            "days": scenario["days"], "end_offset_days": scenario["end_offset_days"],
            "requested_start_utc": result["requested_start_utc"], "requested_end_utc": result["requested_end_utc"],
            "entry_timeframe": result["entry_timeframe"], "execution_timeframe": result.get("execution_timeframe", result["entry_timeframe"]),
            "fee_bps_per_side": scenario["fee_bps"], "slippage_bps_per_side": scenario["slippage_bps"],
            "session_variant": scenario["sessions"], "body_ratio": scenario["body_ratio"], "pivot_lookback": scenario["pivot_lookback"],
            "session_clock": scenario["session_clock"],
            "require_origin_sweep": scenario["require_origin_sweep"],
            "observed_entry_bars": result["observed_entry_bars"], "coverage_percent": result["entry_coverage_percent"],
            "signals": result["signals"], "completed_trades": result["trades_count"],
            "long_trades": result["long_trades"], "short_trades": result["short_trades"],
            "unfilled_signals": result["unfilled_signals"], "expired_limits": result.get("expired_limits"),
            "unfinished_position": result["unfinished_position"], "unfinished_limit": result["unfinished_limit"],
            "wins": result["wins"], "win_rate_percent": result["win_rate"],
            "net_pnl_usdt": result["net_pnl_usdt"], "return_percent": result["return_percent"],
            "profit_factor": result["profit_factor"], "max_realized_drawdown_percent": result["max_realized_drawdown_percent"],
            "final_realized_equity": result["final_realized_equity"], "elapsed_seconds": result["elapsed_seconds"],
            "funding_data_available": result["funding_data_available"], "funding_total_usdt": result.get("funding_total_usdt"),
            "funding_events_charged": result.get("funding_events_charged"),
            "funding_events_skipped_favorable": result.get("funding_events_skipped_favorable"),
            "unrealized_pnl_usdt": result.get("unrealized_pnl_usdt"), "marked_equity": result.get("marked_equity"),
            "max_marked_drawdown_percent": result.get("max_marked_drawdown_percent"),
            "ambiguous_expiry_limits": result.get("ambiguous_expiry_limits"),
        })
    return rows


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def make_report(report):
    results = report["results"]
    baseline = [r for r in results if r["scenario"]["category"] == "baseline"]
    positive = sum(r["net_pnl_usdt"] > 0 for r in baseline)
    negative = sum(r["net_pnl_usdt"] < 0 for r in baseline)
    without_trades = sum(r["trades_count"] == 0 for r in baseline)
    small = sum(r["trades_count"] < 30 for r in baseline)
    expected = len(report["scenario_plan"]) * len(report["symbols"])
    provenance = report.get("universe_provenance")
    timing_note = []
    if provenance:
        timing_note = [
            f"Время выбора текущего списка — {provenance['selected_at_utc']}; оно на "
            f"{provenance['selection_after_ohlc_cutoff_seconds'] / 60:.2f} минуты позже "
            f"фиксированного OHLC-снимка ({provenance['ohlc_cutoff_utc']}). Поэтому состав пар "
            "определён ретроспективно даже относительно конца исторического окна и не является "
            "правилом выбора, которое было известно на тот момент. Историческую доходность "
            "динамической ротации эти эксперименты не измеряют.", ""]
    lines = [
        "# Проверка стратегии из видео на истории MEXC",
        "",
        f"Источник стратегии: [видео]({VIDEO_URL}). Снимок сервера: {report['snapshot']['server_snapshot_utc']}.",
        "",
        "Первичный список пар: " + ", ".join(report["symbols"]) + ". Список выбран по текущей капитализации и волатильности "
        "и затем зафиксирован на всей истории. Это не историческая динамическая ротация top-cap/top-volatility; "
        "выбор текущих выживших активов создаёт смещение состава выборки.",
        "",
        *timing_note,
        f"Завершено {len(results)} из {expected} экспериментов. "
        + ("Полный набор расчётов завершён." if report.get("completed_at_utc") else "Это промежуточный отчёт; вычисления продолжаются."),
        "",
        f"Из {len(baseline)} завершённых базовых экспериментов {positive} имеют положительный PnL, "
        f"{negative} — отрицательный; в {without_trades} нет закрытых сделок. В {small} экспериментах меньше 30 "
        "закрытых сделок. Это описательный порог для отметки малой выборки, а не доказательство статистической достаточности. "
        "Результаты не подтверждают обещанный процент выигрышей или пригодность для реальной торговли.",
        "",
        "Параметры были зафиксированы до вычисления результатов: отдельный счёт 1 000 USDT для каждой пары, "
        "риск 0,5% на сделку, плечо ограничено 3, дневной лимит потерь 2%. Базовые издержки: комиссия 5 bps "
        "и проскальзывание 2 bps на каждой стороне; стресс: 10 + 5 bps на стороне. Это независимые эксперименты, "
        "их прибыли нельзя складывать как доход одного портфеля.",
        "",
        "Базовые часы в UTC+3: 10:00–12:00 и 16:30–18:00. Чувствительность проверяет только утреннее окно, "
        "минимальное отношение тела свечи 0,5/0,7 вместо 0,6 и подтверждение экстремумов по 3 свечам вместо 2. "
        "Структурные стоп/цель сохраняются, лимитная цена рассчитана для 2R. Эти варианты не выбирались по прибыльности.",
        "",
        "DEVIATION_origin_sweep_off — отдельная проверка с отключённым обязательным дневным sweep у истока движения. "
        "Это отклонение от базовой интерпретации видео, не рекомендуемый или подобранный по прибыли вариант.",
        "",
        "HYPOTHESIS_market_local — гипотеза: London 08:00–10:00 Europe/London и New York 09:30–11:00 "
        "America/New_York с историческим DST. [Описание авторского индикатора](https://www.tradingview.com/script/1WFDkBkJ-Trading-Sessions-by-klmn1k-from-Trading-Volium/) "
        "подтверждает учёт DST и вариант открытия New York 09:30, но не устанавливает London 08:00 или точные торговые kill zones. "
        "Эти часы не выдаются за проверенные настройки автора. Версия morning оставляет только первое рыночное окно.",
        "",
        "Используются только закрытые свечи. OHLC не раскрывает последовательность цен внутри свечи: "
        "при касании стопа и цели приоритет получает стоп; благоприятная цель на свече неопределённого исполнения "
        "лимита не считается подтверждённой. Незакрытые позиции исключены из реализованной прибыли/просадки; "
        "отдельные unrealized/marked-поля в JSON и CSV показывают переоценку и просадку по доступным OHLC. "
        "Учитываются фактические публичные ставки funding по времени расчёта; стоимость позиции оценивается по "
        "доступному закрытию базового актива, а не по историческому mark price. Наличие данных funding указано ниже. "
        "Очередь исполнения, изменение контрактных правил и ликвидация не моделируются. "
        "Сценарии с исполнением по 1m уточняют последние 30 дней; остальные исполняются по свечам входного таймфрейма.",
        "",
        "## Полнота исходных данных",
        "",
        "| Пара | TF | Свечей | Первая UTC | Последнее закрытие UTC | Пропусков |",
        "|---|---|---:|---|---|---:|",
    ]
    for symbol in report["symbols"]:
        frames = report["snapshot"]["frames"][symbol]
        for tf, info in frames.items():
            lines.append(f"| {symbol} | {tf} | {info['bars']} | {info['first_open_utc']} | {info['last_close_utc']} | {info['missing_intervals']} |")
    lines += ["", "Funding: [публичная история MEXC](https://www.mexc.com/api-docs/futures/market-endpoints/get-funding-rate-history).", ""]
    for symbol in report["symbols"]:
        info = report["snapshot"].get("funding", {}).get(symbol, {"status": "unavailable", "reason": "No data"})
        if info["status"] == "available":
            lines.append(f"- {symbol}: {info['settlements']} расчётов, {info['first_settlement_utc']} – {info['last_settlement_utc']}, "
                         f"максимальный интервал {number(info['maximum_gap_hours'])} ч.")
        else:
            lines.append(f"- {symbol}: funding недоступен; он исключён из результатов этой пары. Причина: {info['reason']}.")
    lines += ["", "## Базовые результаты", "",
              "| Пара | Режим | Дней | Сделок | LONG / SHORT | Незаполненные сигналы | Win % | PnL USDT | Доход % | PF | Просадка % |",
              "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        if result["scenario"]["category"] != "baseline":
            continue
        scenario = result["scenario"]
        label = "swing W1/H4" if scenario["swing_context"] == "1w" else result["mode"]
        lines.append(f"| {result['symbol']} | {label} | {scenario['days']} | {result['trades_count']} | "
            f"{result['long_trades']} / {result['short_trades']} | {result['unfilled_signals']} | "
            f"{number(result['win_rate'])} | {number(result['net_pnl_usdt'])} | {number(result['return_percent'])} | "
            f"{number(result['profit_factor'])} | {number(result['max_realized_drawdown_percent'])} |")
    lines += ["", "## Все эксперименты", "",
              "Окна overlap пересекаются и не являются независимыми подтверждениями. Segment — три непересекающихся "
              "отрезка по 60 дней для intraday, 10 для scalp и 120 для swing; каждый начинает с собственного капитала 1 000 USDT. "
              "Для swing три отрезка покрывают последние 360 из 365 дней.",
              "",
              "| Сценарий | Пара | Окно UTC | Часы | Издержки bps/сторона | Баров | Покрытие % | Сделок L/S | Незаполн. | Win % | PnL USDT | DD % |",
              "|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for result in results:
        scenario = result["scenario"]
        window = f"{result['requested_start_utc'][:10]}–{result['requested_end_utc'][:10]}"
        lines.append(f"| {scenario['name']} | {result['symbol']} | {window} | {scenario['session_clock']} / {scenario['sessions']} | {scenario['fee_bps']}+{scenario['slippage_bps']} | "
            f"{result['observed_entry_bars']} | {number(result['entry_coverage_percent'])} | "
            f"{result['trades_count']} ({result['long_trades']}/{result['short_trades']}) | {result['unfilled_signals']} | "
            f"{number(result['win_rate'])} | {number(result['net_pnl_usdt'])} | {number(result['max_realized_drawdown_percent'])} |")
    lines += ["", "Незаполненные сигналы включают истёкшие лимитные заявки и заявку, остающуюся неисполненной в конце окна. "
              "ambiguous_expiry_limits отдельно считает заявки, срок которых истёк внутри грубой свечи и исполнение которых "
              "невозможно подтвердить по OHLC; они не превращаются в вымышленные выигрыши. "
              "Количество сделок и результаты, включая отрицательные, приведены без отбора. Малое число сделок ограничивает "
              "силу вывода; исторический процент выигрышей не является обещанием результата.",
              "", "Полные параметры, сделки и UTC-времена находятся в backtest_suite.json; машинная сводка — backtest_summary.csv.",
              ""]
    return "\n".join(lines)


def write_outputs(out_dir, report):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backtest_suite.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows(report["results"])).to_csv(out_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "report.md").write_text(make_report(report), encoding="utf-8")


def asset_worker(symbol, full_frames, scenarios, snapshot, funding_rates, progress_path):
    """Pure computation worker; public network access occurs only in the parent fetch stage."""
    signal_cache = SignalCache()
    results = []
    target = Path(progress_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    for index, scenario in enumerate(scenarios, 1):
        print(f"[{symbol} {index}/{len(scenarios)}] {scenario.name}", flush=True)
        result = run_case(symbol, full_frames, scenario, snapshot, signal_cache, funding_rates)
        results.append(result)
        payload = {"symbol": symbol, "results": results,
                   "signal_cache": signal_cache.metrics()}
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
        print(f"  {symbol}: trades={result['trades_count']} L/S={result['long_trades']}/{result['short_trades']} "
              f"unfilled={result['unfilled_signals']} pnl={result['net_pnl_usdt']:.2f}USDT "
              f"win={number(result['win_rate'])}% elapsed={result['elapsed_seconds']:.1f}s", flush=True)
    return payload


def merge_completed_payloads(payloads, scenarios, symbols):
    """Gather authoritative worker returns and verify the entire planned suite."""
    expected = {(scenario.name, symbol) for scenario in scenarios for symbol in symbols}
    if len(expected) != len(scenarios) * len(symbols):
        raise RuntimeError("Duplicate scenario or symbol in suite plan")
    by_case = {}
    totals = {"hits": 0, "misses": 0, "broad_rejections": 0, "broad_evaluations": 0}
    for payload in payloads:
        for metric in totals:
            totals[metric] += payload["signal_cache"][metric]
        for result in payload["results"]:
            key = (result["scenario"]["name"], result["symbol"])
            if key in by_case and by_case[key] != result:
                raise RuntimeError(f"Conflicting worker results for {key}")
            by_case[key] = result
    missing, unexpected = expected - by_case.keys(), by_case.keys() - expected
    if missing or unexpected or len(by_case) != len(expected):
        raise RuntimeError(f"Incomplete suite: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    order = {scenario.name: index for index, scenario in enumerate(scenarios)}
    ordered = sorted(by_case.values(), key=lambda result: (
        order[result["scenario"]["name"]], symbols.index(result["symbol"])))
    return ordered, totals


async def run(args):
    universe = json.loads(Path(args.universe_json).read_text(encoding="utf-8")) if args.universe_json else None
    supplied_symbols = args.symbol
    if supplied_symbols is None and universe:
        supplied_symbols = universe.get("symbols") or [row["symbol"] for row in universe.get("selected", [])]
    supplied_symbols = supplied_symbols or list(SYMBOLS)
    symbols = Settings(_env_file=None, trading_symbols=supplied_symbols).trading_symbols
    scenarios = build_scenarios(not args.skip_weekly, not args.skip_fine)
    if args.scenario:
        scenarios = [s for s in scenarios if s.name in args.scenario]
        if not scenarios:
            raise ValueError("No matching scenarios")
    if args.dry_run:
        print(json.dumps({"symbols": symbols, "public_cache_limits": CACHE_LIMITS,
                          "cases": len(scenarios) * len(symbols),
                          "scenarios": [asdict(s) for s in scenarios]}, indent=2))
        return
    cache_dir, out_dir = Path(args.cache), Path(args.out_dir)
    frames, snapshot = await fetch_snapshot(cache_dir, symbols, args.refresh_cache, args.cache_only)
    funding = await funding_snapshot(cache_dir, snapshot, symbols, args.cache_only)
    if args.fetch_only:
        print("Public cache snapshot complete; no replay was run.", flush=True)
        return
    report = {"strategy_video": VIDEO_URL, "snapshot": snapshot,
              "symbols": symbols, "universe_selection": universe,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "frozen_assumptions": {
                  "equity_per_symbol_usdt": 1000, "risk_percent": 0.5, "leverage_cap": 3,
                  "daily_loss_percent": 2, "baseline_fee_bps_per_side": 5, "baseline_slippage_bps_per_side": 2,
                  "stress_fee_bps_per_side": 10, "stress_slippage_bps_per_side": 5,
                  "baseline_sessions_utc3": BASELINE_SESSIONS, "independent_symbol_accounts": True,
                  "no_parameter_selection_for_profit": True, "unrealized_pnl_excluded_from_realized_metrics": True,
                  "marked_equity_and_drawdown_reported_separately": True,
                  "one_open_signal_per_symbol": True, "actual_funding_rates": True,
                  "funding_position_value_uses_underlying_closed_price": True,
                  "liquidation_order_queue_excluded": True},
              "scenario_plan": [asdict(s) for s in scenarios], "results": []}
    if universe and universe.get("selected_at"):
        selected_at = universe["selected_at"]
        cutoff = snapshot["server_snapshot_utc"]
        report["universe_provenance"] = {
            "selected_at_utc": selected_at, "ohlc_cutoff_utc": cutoff,
            "selection_after_ohlc_cutoff_seconds": (
                datetime.fromisoformat(selected_at) - datetime.fromisoformat(cutoff)).total_seconds(),
            "historical_dynamic_membership": False,
            "bias_note": "The later current universe is fixed over the whole earlier historical period; "
                         "this is retrospective composition/survivorship selection, not an as-of historical portfolio rule."}
    report["code_sha256"] = {name: sha256_file(Path(__file__).parent / name)
                             for name in ("backtest_suite.py", "replay.py", "strategy/volium.py", "config.py")}
    report["settings_by_scenario"] = {s.name: s.settings().model_dump() for s in scenarios}
    signal_cache = SignalCache()
    total = len(scenarios) * len(symbols)
    if args.workers == 1 or len(symbols) == 1:
        for scenario in scenarios:
            for symbol in symbols:
                index = len(report["results"]) + 1
                print(f"[{index}/{total}] {symbol} {scenario.name}", flush=True)
                result = run_case(symbol, frames[symbol], scenario,
                                  datetime.fromisoformat(snapshot["server_snapshot_utc"]), signal_cache, funding[symbol])
                report["results"].append(result)
                report["signal_cache"] = signal_cache.metrics()
                write_outputs(out_dir, report)
                print(f"  trades={result['trades_count']} L/S={result['long_trades']}/{result['short_trades']} "
                      f"unfilled={result['unfilled_signals']} pnl={result['net_pnl_usdt']:.2f}USDT "
                      f"win={number(result['win_rate'])}% elapsed={result['elapsed_seconds']:.1f}s", flush=True)
    else:
        progress_dir = cache_dir / "progress" / str(time.time_ns())
        progress_dir.mkdir(parents=True, exist_ok=True)
        order = {s.name: index for index, s in enumerate(scenarios)}
        snapshot_at = datetime.fromisoformat(snapshot["server_snapshot_utc"])
        groups = list(dict.fromkeys((s.mode, s.swing_context) for s in scenarios))
        jobs = [(f"{symbol}_{mode}_{context}", symbol, [s for s in scenarios if (s.mode, s.swing_context) == (mode, context)])
                for mode, context in groups for symbol in symbols]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(symbols))) as pool:
            futures = {job: pool.submit(asset_worker, symbol, frames[symbol], job_scenarios, snapshot_at,
                                       funding[symbol], str(progress_dir / f"{job}.json"))
                       for job, symbol, job_scenarios in jobs}
            previous_count = -1
            while True:
                checkpoint_results, totals = [], {"hits": 0, "misses": 0, "broad_rejections": 0, "broad_evaluations": 0}
                for job, _, _ in jobs:
                    path = progress_dir / f"{job}.json"
                    if path.exists():
                        checkpoint = json.loads(path.read_text(encoding="utf-8"))
                        checkpoint_results.extend(checkpoint["results"])
                        for key in totals:
                            totals[key] += checkpoint["signal_cache"][key]
                if len(checkpoint_results) != previous_count:
                    report["results"] = sorted(checkpoint_results, key=lambda r: (order[r["scenario"]["name"]], symbols.index(r["symbol"])))
                    report["signal_cache"] = totals
                    write_outputs(out_dir, report)
                    previous_count = len(checkpoint_results)
                    print(f"Checkpoint: {previous_count}/{total} cases complete", flush=True)
                if all(future.done() for future in futures.values()):
                    # A worker may finish after its checkpoint was read above.
                    # Its return value is authoritative, so collect it before
                    # marking the report complete instead of trusting a stale file.
                    payloads = [future.result() for future in futures.values()]
                    report["results"], report["signal_cache"] = merge_completed_payloads(payloads, scenarios, symbols)
                    break
                # Keep the event loop responsive while independent assets compute.
                await asyncio.sleep(5)
    report["results"], report["signal_cache"] = merge_completed_payloads(
        [{"results": report["results"], "signal_cache": report["signal_cache"]}], scenarios, symbols)
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_outputs(out_dir, report)
    print(f"Completed {total} cases. Outputs: {out_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=".cache/suite")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUTS))
    parser.add_argument("--dry-run", action="store_true", help="Show all parameters and data requirements without network access")
    parser.add_argument("--fetch-only", action="store_true", help="Fetch and freeze public candles without replay")
    parser.add_argument("--cache-only", action="store_true", help="Never use the network; require a complete verified cache")
    parser.add_argument("--refresh-cache", action="store_true", help="Create a new frozen public snapshot")
    parser.add_argument("--skip-weekly", action="store_true")
    parser.add_argument("--skip-fine", action="store_true")
    parser.add_argument("--scenario", action="append", help="Run a named planned scenario (repeatable)")
    parser.add_argument("--symbol", action="append", help="Fixed contract universe; repeat for each asset")
    parser.add_argument("--universe-json", help="Current-universe selection evidence with symbols or selected rows")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2, help="Independent public-data replay processes")
    asyncio.run(run(parser.parse_args()))
