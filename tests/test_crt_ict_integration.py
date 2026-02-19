from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from app.config import Settings
from app.schemas.common import TrendDirection
from app.strategy.crt import CRTCandle
from app.strategy.crt_ict import analyze_crt_ict_from_df


def _ltf_frame() -> pd.DataFrame:
    idx = pd.date_range(datetime(2025, 1, 1, tzinfo=timezone.utc), periods=80, freq="min")
    base = pd.DataFrame(
        {
            "open": [100.0] * len(idx),
            "high": [100.6] * len(idx),
            "low": [99.6] * len(idx),
            "close": [100.0] * len(idx),
        },
        index=idx,
    )
    return base


def _htf_frame(bullish: bool = True) -> pd.DataFrame:
    idx = pd.date_range(datetime(2025, 1, 1, tzinfo=timezone.utc), periods=80, freq="h")
    sign = 1.0 if bullish else -1.0
    close = [sign * (100 + i * 0.1) for i in range(len(idx))]
    frame = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)
    return frame


def _mock_crt(ltf_df: pd.DataFrame, direction: str = "LONG") -> CRTCandle:
    return CRTCandle(
        timestamp=ltf_df.index[-1].to_pydatetime(),
        direction=direction,  # type: ignore[arg-type]
        range_high=101.0,
        range_low=99.0,
        sweep_level=98.5 if direction == "LONG" else 101.5,
        close_inside=True,
        rejection_ratio=2.0,
    )


def test_crt_ict_no_signal_when_htf_disagree(monkeypatch):
    settings = Settings()
    ltf = _ltf_frame()
    htf_4h = _htf_frame(bullish=True)
    htf_1d = _htf_frame(bullish=False)

    monkeypatch.setattr(
        "app.strategy.crt_ict.analyze_htf_from_df",
        lambda df: SimpleNamespace(bias=TrendDirection.BULLISH if float(df["close"].mean()) > 0 else TrendDirection.BEARISH),
    )
    monkeypatch.setattr("app.strategy.crt_ict.detect_crt_candle", lambda *_args, **_kwargs: _mock_crt(ltf, "LONG"))
    monkeypatch.setattr("app.strategy.crt_ict._check_mss_confirmation", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("app.strategy.crt_ict._check_ob_confirmation", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("app.strategy.crt_ict._check_fvg_confirmation", lambda *_args, **_kwargs: True)

    analysis, setup = analyze_crt_ict_from_df("BTC-USDT", ltf, htf_4h, htf_1d, settings)
    assert analysis.signal == "NO_SIGNAL"
    assert setup is None


def test_crt_ict_no_signal_without_enough_confirmations(monkeypatch):
    settings = Settings()
    ltf = _ltf_frame()
    htf_4h = _htf_frame(bullish=True)
    htf_1d = _htf_frame(bullish=True)

    monkeypatch.setattr("app.strategy.crt_ict.analyze_htf_from_df", lambda _df: SimpleNamespace(bias=TrendDirection.BULLISH))
    monkeypatch.setattr("app.strategy.crt_ict.detect_crt_candle", lambda *_args, **_kwargs: _mock_crt(ltf, "LONG"))
    monkeypatch.setattr("app.strategy.crt_ict._check_mss_confirmation", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("app.strategy.crt_ict._check_ob_confirmation", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("app.strategy.crt_ict._check_fvg_confirmation", lambda *_args, **_kwargs: True)

    analysis, setup = analyze_crt_ict_from_df("BTC-USDT", ltf, htf_4h, htf_1d, settings)
    assert analysis.signal == "NO_SIGNAL"
    assert setup is None


def test_crt_ict_targets_use_liquidity_then_rr_fallback(monkeypatch):
    settings = Settings()
    settings.crt_killzone_enabled = False
    ltf = _ltf_frame()
    htf_4h = _htf_frame(bullish=True)
    htf_1d = _htf_frame(bullish=True)

    monkeypatch.setattr("app.strategy.crt_ict.analyze_htf_from_df", lambda _df: SimpleNamespace(bias=TrendDirection.BULLISH))
    monkeypatch.setattr("app.strategy.crt_ict.detect_crt_candle", lambda *_args, **_kwargs: _mock_crt(ltf, "LONG"))
    monkeypatch.setattr("app.strategy.crt_ict._check_mss_confirmation", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("app.strategy.crt_ict._check_ob_confirmation", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("app.strategy.crt_ict._check_fvg_confirmation", lambda *_args, **_kwargs: False)

    monkeypatch.setattr(
        "app.strategy.crt_ict._extract_liquidity_levels",
        lambda *_args, **_kwargs: ([105.0], [95.0], 105.0, 95.0),
    )
    analysis_liq, setup_liq = analyze_crt_ict_from_df("BTC-USDT", ltf, htf_4h, htf_1d, settings)
    assert analysis_liq.signal == "LONG"
    assert setup_liq is not None
    assert round(setup_liq.take_profits[0], 6) == 105.0

    monkeypatch.setattr(
        "app.strategy.crt_ict._extract_liquidity_levels",
        lambda *_args, **_kwargs: ([], [], None, None),
    )
    analysis_rr, setup_rr = analyze_crt_ict_from_df("BTC-USDT", ltf, htf_4h, htf_1d, settings)
    assert analysis_rr.signal == "LONG"
    assert setup_rr is not None
    risk = abs(setup_rr.entry - setup_rr.stop_loss)
    assert round(setup_rr.take_profits[0], 6) == round(setup_rr.entry + risk * 2, 6)
