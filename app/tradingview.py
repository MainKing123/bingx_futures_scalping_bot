from __future__ import annotations

TIMEFRAME_TO_TV = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "8h": "480",
    "12h": "720",
    "1d": "D",
    "1w": "W",
}


def symbol_to_tv_candidates(symbol: str) -> list[str]:
    base_quote = "".join(ch for ch in symbol.upper() if ch.isalnum())
    return [
        f"BINGX:{base_quote}.P",
        f"BINGX:{base_quote}",
        f"BINANCE:{base_quote}.P",
    ]


def timeframe_to_tv_interval(interval: str) -> str:
    return TIMEFRAME_TO_TV.get(interval, "5")
