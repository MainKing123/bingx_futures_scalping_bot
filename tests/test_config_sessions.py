from app.config import SessionConfig, is_active_session


def test_all_sessions_flag_disables_time_filter():
    assert is_active_session(SessionConfig(enabled=["all"])) is True
