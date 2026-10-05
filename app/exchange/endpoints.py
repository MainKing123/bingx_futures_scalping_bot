"""MEXC futures endpoints verified against the current official API docs."""

BASE_URL = "https://api.mexc.com"
WS_MARKET_URL = "wss://contract.mexc.com/edge"
SERVER_TIME = "/api/v1/contract/ping"
CONTRACTS = "/api/v1/contract/detail/country"
KLINES = "/api/v1/contract/kline/{symbol}"
TICKER = "/api/v1/contract/ticker"
DEPTH = "/api/v1/contract/depth/{symbol}"
FAIR_PRICE = "/api/v1/contract/fair_price/{symbol}"
BALANCE = "/api/v1/private/account/asset/{currency}"
POSITIONS = "/api/v1/private/position/open_positions"
HISTORY_POSITIONS = "/api/v1/private/position/list/history_positions"
POSITION_MODE = "/api/v1/private/position/position_mode"
ORDER = "/api/v1/private/order/create"
ORDER_BY_ID = "/api/v1/private/order/get/{order_id}"
ORDER_BY_EXTERNAL_ID = "/api/v1/private/order/external/{symbol}/{external_oid}"
OPEN_ORDERS = "/api/v1/private/order/list/open_orders"
CANCEL = "/api/v1/private/order/cancel"
LEVERAGE = "/api/v1/private/position/change_leverage"
STOP_ORDERS = "/api/v1/private/stoporder/open_orders"

INTERVALS = {
    "1m": ("Min1", 60), "5m": ("Min5", 300), "15m": ("Min15", 900),
    "30m": ("Min30", 1800), "1h": ("Min60", 3600), "4h": ("Hour4", 14400),
    "8h": ("Hour8", 28800), "1d": ("Day1", 86400),
    "1w": ("Week1", 604800), "1M": ("Month1", None),
}


def normalize_interval(interval: str) -> str:
    if interval in INTERVALS:
        return interval
    for label, (api_interval, _) in INTERVALS.items():
        if interval == api_interval:
            return label
    raise ValueError(f"Unsupported MEXC futures interval: {interval}")
