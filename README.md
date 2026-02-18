# BingX Futures Scalping Bot (MVP)

MVP implementation of an ICT/SMC-inspired scalping bot skeleton with:
- FastAPI backend
- Strategy engine (30m context + 1m entry signal simulation)
- Risk and position sizing logic (leverage, TP/SL)
- Position management (TP1 partial close, breakeven, trailing stop)
- Risk guards (daily loss limit, max consecutive losses, cooldown)
- Minimal web UI dashboard

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000

## API
- `GET /api/state` bot snapshot
- `POST /api/risk-config` update risk settings
- `POST /api/strategy-config` update strategy thresholds
- `POST /api/tick` push a synthetic tick and evaluate strategy/position lifecycle
- `POST /api/close-position` force-close active position (`{"close_price": <number>}`)
- `POST /api/reset` reset runtime bot state (`{"keep_configs": true|false}`)
