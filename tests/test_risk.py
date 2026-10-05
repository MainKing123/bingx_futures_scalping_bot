from app.config import Settings
from app.risk.risk_manager import RiskManager


def test_position_size_positive():
    rm = RiskManager(Settings())
    size = rm.calculate_position_size(entry=100, stop_loss=99, risk_percent=1, balance=1000)
    assert size > 0
