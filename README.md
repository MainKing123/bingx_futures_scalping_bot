# BingX Futures Scalping Bot

FastAPI backend for scanning BingX futures pairs with SMC/ICT logic, Strategy Lab backtests, and a web dashboard.

## What is implemented

- Live scanner for top volatile USDT pairs (default top-20)
- Legacy setup flow (`CHOCH_OB`)
- New `CRT+ICT` strategy:
  - HTF bias from `1D + 4H`
  - Entry timeframes `5m/15m`
  - Killzone filter (toggleable)
  - Structured analysis output
- Strategy Lab backtest:
  - `single` and `batch`
  - `strategy=crt_ict` and `strategy=legacy_choch_ob`
  - KPI: `expectancy`, `profit_factor`, `win_rate`, `max_drawdown`, `trades_count`
- TradingView panel in dashboard with symbol fallback:
  - `BINGX:{SYMBOL}.P` -> `BINGX:{SYMBOL}` -> `BINANCE:{SYMBOL}.P`

## Main API routes

- `GET /api/setups`
- `GET /api/watchlist`
- `GET /api/scanner/volatile-pairs`
- `GET /api/analysis/{symbol}`
- `POST /api/backtest/run`
- `GET /api/backtest/jobs/{job_id}`
- `GET /api/backtest/jobs/{job_id}/result`
- `GET /api/backtest/latest`
- `GET /api/tradingview/symbol/{symbol}`
- `GET /api/config`
- `PATCH /api/config`

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Dashboard: `http://127.0.0.1:8000/dashboard`
- API docs: `http://127.0.0.1:8000/docs`

## Configuration highlights

New CRT/rollout settings in `.env`:

- `STRATEGY_LIVE_MODE=crt_shadow` (`legacy`, `crt_shadow`, `crt_live`)
- `CRT_KILLZONE_ENABLED=true`
- `CRT_LONDON_SESSION=[2,5]`
- `CRT_NEW_YORK_SESSION=[7,10]`
- `CRT_ENTRY_TIMEFRAMES=["5m","15m"]`
- `CRT_RANGE_LOOKBACK=20`
- `CRT_MIN_SWEEP_PCT=0.03`
- `CRT_MIN_WICK_BODY_RATIO=1.2`
- `CRT_EQUAL_LEVEL_TOLERANCE=0.0005`
- `CRT_MSS_LOOKBACK=8`
- `CRT_STOP_BUFFER_BPS=2.0`
- `CRT_MIN_RR=2.0`

## Notes

- Public market data endpoints work without exchange API keys for scanner/backtest/analysis.
- API keys are required only for private account actions (balance, positions, order execution).
