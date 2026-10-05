"""Render a causal multi-timeframe research setup for manual source review.

Only information closed at the signal is displayed. Outcome prices, final-test
selection and parameter optimization are not part of this utility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from app.config import Settings
from app.replay import SECONDS, required_timeframes
from app.strategy.volium import _pivots
from app.strategy.volium_v2 import V2Parameters, diagnose_volium_v2_from_df


def write_review(cache, output, *, symbol, mode, signal_time, parameters=None, strategy_version="v2"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cache, output = Path(cache), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if strategy_version == "v4":
        from app.strategy.volium_v4 import V4Parameters, diagnose_volium_v4_from_df
        parameters = parameters or V4Parameters()
        diagnose = diagnose_volium_v4_from_df
    elif strategy_version == "v3":
        from app.strategy.volium_v3 import V3Parameters, diagnose_volium_v3_from_df
        parameters = parameters or V3Parameters()
        diagnose = diagnose_volium_v3_from_df
    elif strategy_version == "v2":
        parameters = parameters or V2Parameters()
        diagnose = diagnose_volium_v2_from_df
    else:
        raise ValueError("Review strategy version must be explicitly v2, v3 or v4")
    current = pd.Timestamp(signal_time)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    manifest_path = cache / "snapshot.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = Settings(_env_file=None, mexc_api_key="", mexc_api_secret="", auto_execution=False,
                        volium_mode=mode, volium_context_lookback=80,
                        volium_session_enabled=True, volium_session_clock="fixed_utc3",
                        volium_sessions_utc3=[("10:00", "12:00"), ("16:30", "18:00")])
    frames, hashes = {}, {}
    for tf in required_timeframes(settings):
        record = manifest["frames"][symbol][tf]
        path = cache / record.get("filename", f"{symbol}_{tf}.csv")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Modified source data: {path.name}")
        frame = pd.read_csv(path, index_col=0)
        frame.index = pd.to_datetime(frame.index, utc=True)
        history = settings.volium_context_lookback*24+110 if strategy_version in {"v3","v4"} and tf == "1h" else 110
        frame = frame.loc[frame.index + pd.Timedelta(seconds=SECONDS[tf]) <= current].tail(history)
        if frame.empty or frame.index.has_duplicates:
            raise ValueError("Review needs complete causal frames")
        frames[tf], hashes[path.name] = frame, digest
    diagnostic = diagnose(symbol=symbol, frames=frames, settings=settings,
                                             mode=mode, now=current, parameters=parameters)
    if diagnostic.setup is None:
        raise ValueError(f"No accepted setup at this confirmed close: {diagnostic.reason}")
    setup, features = diagnostic.setup, diagnostic.features
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True)
    fig.patch.set_facecolor("#f7f9fc")
    venue = manifest.get("venue", "mexc_futures")
    fig.suptitle(f"{venue} · {strategy_version} · {symbol} · {mode} · {setup.direction}\n"
                 f"Закрытое подтверждение: {current.isoformat()} | Только известные к этому времени свечи",
                 fontsize=15, fontweight="bold")
    timeframes = required_timeframes(settings)
    for row, (ax, tf) in enumerate(zip(axes, timeframes)):
        frame = frames[tf].tail(80 if row < 2 else 45)
        ax.set_facecolor("white")
        for i, candle in enumerate(frame.itertuples()):
            color = "#1b9975" if candle.close >= candle.open else "#d45d66"
            ax.vlines(i, candle.low, candle.high, color=color, linewidth=1)
            height = abs(candle.close-candle.open)
            height = max(height, float(frame.high.max()-frame.low.min()) * .0007)
            ax.add_patch(Rectangle((i-.3, min(candle.open,candle.close)), .6, height,
                                    facecolor=color, edgecolor=color))
        known_pivots = _pivots(frame, settings.volium_swing_lookback)
        for point in known_pivots:
            ax.scatter(point.index, point.price, marker="v" if point.kind == "high" else "^",
                       s=16, color="#59687a", zorder=4)
        ticks = list(range(0,len(frame),max(1,len(frame)//7)))
        if ticks and len(frame)-1-ticks[-1] < max(1,len(frame)//14):
            ticks.pop()
        if len(frame)-1 not in ticks:
            ticks.append(len(frame)-1)
        ax.set_xticks(ticks, [frame.index[i].strftime("%d %b\n%H:%M") for i in ticks])
        ax.set_xlim(-1,len(frame))
        ax.grid(axis="y", color="#e7ebf0", linewidth=.7)
        ax.set_ylabel("Цена (USDT)")
        ax.set_title([f"{tf}: активное движение и тренд" if mode == "scalp" else f"{tf}: контекст и ещё неснятая цель B",
                      f"{tf}: ликвидность и цель B" if mode == "scalp" else f"{tf}: ликвидность и начало коррекции",
                      f"{tf}: снятие → V-реакция → лимитный ретест 2R"][row], loc="left", fontweight="bold")
        annotations = []
        if row == 0 and "context_origin_open_utc" in features:
            annotations += [(features["context_origin_open_utc"], features["context_origin_extreme"], "A: край снятия", "#8963b0")]
            annotations += [(features["context_target_open_utc"], features["context_target"], "B: дневная ликвидность", "#377eb8")]
            confirmed_day = features.get("context_confirmation_open_utc")
            if confirmed_day is not None and pd.Timestamp(confirmed_day) in frame.index:
                at = frame.index.get_loc(pd.Timestamp(confirmed_day))
                value = features["context_confirmation_close"]
                ax.scatter(at,value,marker="o",s=32,color="#1b9975",zorder=5)
                ax.annotate("C: закрытое подтверждение ноги",(at,value),xytext=(5,-16),
                            textcoords="offset points",color="#1b9975",fontsize=9)
        if row == 1:
            annotations += [(features["liquidity_level_open_utc"], features["liquidity_level"], "Снятая ликвидность", "#8963b0")]
            if features.get("correction_origin_open_utc"):
                annotations += [(features["correction_origin_open_utc"], setup.take_profits[0], "Начало коррекции / TP", "#377eb8")]
            elif mode == "scalp":
                ax.axhline(setup.take_profits[0],color="#377eb8",linestyle="--",alpha=.7,linewidth=1)
                ax.text(.01,setup.take_profits[0],"B / структурный TP",transform=ax.get_yaxis_transform(),
                        color="#377eb8",fontsize=9)
        for opened, price, label, color in annotations:
            timestamp = pd.Timestamp(opened)
            ax.axhline(price, color=color, linestyle="--", alpha=.7, linewidth=1)
            if timestamp in frame.index:
                at = frame.index.get_loc(timestamp)
                ax.annotate(label, (at,price), xytext=(5,10), textcoords="offset points", color=color, fontsize=9)
            else:
                ax.text(.01,price,label,transform=ax.get_yaxis_transform(),color=color,fontsize=9)
        if row == 2:
            for price, label, color in [(setup.stop_loss,"SL за снятием","#d45d66"),
                                        (setup.entry,"Лимитный вход","#8963b0"),
                                        (setup.take_profits[0],"Структурный TP","#1b9975")]:
                ax.axhline(price, color=color, linestyle="--", linewidth=1)
                ax.text(.01,price,label,transform=ax.get_yaxis_transform(),color=color,fontsize=9,
                        va="bottom",bbox={"facecolor":"white","edgecolor":"none","alpha":.8,"pad":1})
            sweep_time = pd.Timestamp(features["sweep_open_utc"])
            if sweep_time in frame.index:
                ax.axvline(frame.index.get_loc(sweep_time), color="#8963b0", alpha=.6, linewidth=1)
            raid_start = features.get("raid_start_open_utc")
            if raid_start is not None and pd.Timestamp(raid_start) in frame.index:
                ax.axvline(frame.index.get_loc(pd.Timestamp(raid_start)), color="#377eb8", alpha=.7,
                           linewidth=1, linestyle=":")
            ax.text(.99,.97,
                    f"Тело / ATR до снятия: {features['confirm_body_atr']:.2f}\n"
                    f"Тело / противоположное тело: {features['confirm_body_prior_body_ratio']:.2f}\n"
                    f"Тело / диапазон: {features['confirm_body_fraction']:.2f}",
                    transform=ax.transAxes,ha="right",va="top",fontsize=9,
                    bbox={"facecolor":"white","edgecolor":"#e7ebf0","alpha":.9})
    stem = f"{venue}_{strategy_version}_{symbol}_{mode}_{current.strftime('%Y%m%d_%H%M')}"
    image_path = output / f"{stem}.png"
    fig.savefig(image_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    evidence = {"venue":venue,"strategy_version":strategy_version,"symbol":symbol,"mode":mode,"signal_time_utc":current.isoformat(),
                "causal_only":True,"outcome_bars_displayed":False,"parameters":asdict(parameters),
                "source_manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "source_sha256":hashes,"diagnostic":diagnostic.as_dict(),"image":image_path.name}
    evidence_path = output / f"{stem}.json"
    evidence_path.write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding="utf-8")
    return image_path,evidence_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--symbol",required=True)
    parser.add_argument("--mode",choices=["intraday","scalp"],required=True)
    parser.add_argument("--signal-time",required=True)
    parser.add_argument("--strategy-version",choices=["v2","v3","v4"],default="v2")
    parser.add_argument("--parameters-json",default="{}")
    args = parser.parse_args()
    parameter_type = V2Parameters
    if args.strategy_version == "v3":
        from app.strategy.volium_v3 import V3Parameters
        parameter_type = V3Parameters
    elif args.strategy_version == "v4":
        from app.strategy.volium_v4 import V4Parameters
        parameter_type = V4Parameters
    paths = write_review(args.cache,args.output,symbol=args.symbol,mode=args.mode,
                         signal_time=args.signal_time,parameters=parameter_type(**json.loads(args.parameters_json)),
                         strategy_version=args.strategy_version)
    print(json.dumps([str(path) for path in paths]))


if __name__ == "__main__":
    main()
