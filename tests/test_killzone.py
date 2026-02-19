from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from app.config import Settings
from app.schemas.common import TrendDirection
from app.strategy.crt import CRTCandle
from app.strategy.crt_ict import analyze_crt_ict_from_df, resolve_killzone


def _frame(size: int = 80) -> pd.DataFrame:
    idx = pd.date_range(datetime(2025, 1, 1, tzinfo=timezone.utc), periods=size, freq="min")
    return pd.DataFrame({"open": 100.0, "high": 100.6, "low": 99.6, "close": 100.0}, index=idx)


def _mock_crt(ts: datetime) -> CRTCandle:
    return CRTCandle(
        timestamp=ts,
        direction="LONG",
        range_high=101.0,
        range_low=99.0,
        sweep_level=98.5,
        close_inside=True,
        rejection_ratio=2.0,
    )


def test_killzone_enabled_blocks_signal_outside_window(monkeypatch):
    settings = Settings()
    settings.crt_killzone_enabled = True
    settings.crt_london_session = (2, 5)
    settings.crt_new_york_session = (7, 10)
    ltf = _frame()
    htf_4h = _frame()
    htf_1d = _frame()
    now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("app.strategy.crt_ict.analyze_htf_from_df", lambda _df: SimpleNamespace(bias=TrendDirection.BULLISH))
    monkeypatch.setattr("app.strategy.crt_ict.detect_crt_candle", lambda *_args, **_kwargs: _mock_crt(now))
    monkeypatch.setattr("app.strategy.crt_ict._check_mss_confirmation", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("app.strategy.crt_ict._check_ob_confirmation", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("app.strategy.crt_ict._check_fvg_confirmation", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("app.strategy.crt_ict._extract_liquidity_levels", lambda *_args, **_kwargs: ([105.0], [95.0], 105.0, 95.0))

    analysis, setup = analyze_crt_ict_from_df("BTC-USDT", ltf, htf_4h, htf_1d, settings, now=now)
    assert analysis.session.killzone_active is False
    assert analysis.signal == "NO_SIGNAL"
    assert setup is None


def test_killzone_disabled_allows_signal(monkeypatch):
    settings = Settings()
    settings.crt_killzone_enabled = False
    ltf = _frame()
    htf_4h = _frame()
    htf_1d = _frame()
    now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("app.strategy.crt_ict.analyze_htf_from_df", lambda _df: SimpleNamespace(bias=TrendDirection.BULLISH))
    monkeypatch.setattr("app.strategy.crt_ict.detect_crt_candle", lambda *_args, **_kwargs: _mock_crt(now))
    monkeypatch.setattr("app.strategy.crt_ict._check_mss_confirmation", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("app.strategy.crt_ict._check_ob_confirmation", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("app.strategy.crt_ict._check_fvg_confirmation", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("app.strategy.crt_ict._extract_liquidity_levels", lambda *_args, **_kwargs: ([105.0], [95.0], 105.0, 95.0))

    analysis, setup = analyze_crt_ict_from_df("BTC-USDT", ltf, htf_4h, htf_1d, settings, now=now)
    assert analysis.session.killzone_active is True
    assert analysis.signal == "LONG"
    assert setup is not None
    assert setup.confidence in {"LOW", "MEDIUM", "HIGH"}


def test_resolve_killzone_optional_toggle():
    settings = Settings()
    settings.crt_killzone_enabled = False
    state = resolve_killzone(datetime(2025, 1, 1, 20, 0, tzinfo=timezone.utc), settings)
    assert state.killzone_enabled is False
    assert state.killzone_active is True
