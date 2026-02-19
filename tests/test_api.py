from fastapi.testclient import TestClient

from app.main import app


def test_health_routes():
    client = TestClient(app)
    assert client.get('/api/stats').status_code == 200
    assert client.get('/api/watchlist').status_code == 200
