# BingX SMC/ICT Backend

Backend-приложение на FastAPI для анализа криптопар BingX и поиска торговых сетапов по SMC/ICT.

## Возможности
- Прямой async-клиент к BingX API (`httpx`) с подписью HMAC-SHA256
- WebSocket стрим свечей BingX (`websockets`, gzip-decompress)
- SMC/ICT анализ: swing points, BOS/CHoCH, OB, FVG, liquidity, premium/discount
- Мультитаймфрейм логика (HTF bias + LTF confirmation)
- Периодический сканер пар через APScheduler
- SQLite + SQLAlchemy 2.0 (async)
- REST API + WebSocket события для клиентов
- Telegram-уведомления через aiogram 3

## Запуск
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Основные API
- `GET /api/setups`
- `GET /api/watchlist`
- `GET /api/market-structure/{symbol}`
- `GET /api/account/balance`
- `GET /api/account/positions`
- `WS /ws`

## Важно
- Добавьте ключи BingX в `.env`
- Для production настройте persistent storage и полноценный error monitoring
