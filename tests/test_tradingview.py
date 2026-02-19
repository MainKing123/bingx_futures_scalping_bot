from app.tradingview import symbol_to_tv_candidates, timeframe_to_tv_interval


def test_symbol_mapping_for_tradingview():
    candidates = symbol_to_tv_candidates("BTC-USDT")
    assert candidates == ["BINGX:BTCUSDT.P", "BINGX:BTCUSDT", "BINANCE:BTCUSDT.P"]


def test_timeframe_mapping_for_tradingview():
    assert timeframe_to_tv_interval("1m") == "1"
    assert timeframe_to_tv_interval("5m") == "5"
    assert timeframe_to_tv_interval("1h") == "60"
