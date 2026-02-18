import unittest

from fastapi.testclient import TestClient

from app.main import app


class ApiWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.client.post('/api/reset', json={'keep_configs': False})

    def test_reset_keep_configs_preserves_custom_risk_values(self) -> None:
        self.client.post(
            '/api/risk-config',
            json={
                'equity_usdt': 1500,
                'risk_per_trade_pct': 0.5,
                'max_leverage': 10,
                'stop_buffer_pct': 0.1,
                'rr_target': 2,
                'daily_loss_limit_pct': 3,
                'max_consecutive_losses': 3,
                'cooldown_minutes': 10,
            },
        )

        response = self.client.post('/api/reset', json={'keep_configs': True})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data['risk_config']['equity_usdt'], 1500)
        self.assertIsNone(data['active_position'])
        self.assertEqual(data['stats']['trades_closed'], 0)

    def test_close_position_accepts_json_payload(self) -> None:
        tick_payload = {
            'symbol': 'BTC-USDT',
            'price': 62000,
            'high_1m': 62100,
            'low_1m': 61900,
            'high_30m': 64000,
            'low_30m': 62000,
        }
        opened = self.client.post('/api/tick', json=tick_payload)
        self.assertEqual(opened.status_code, 200)
        self.assertIsNotNone(opened.json()['active_position'])

        closed = self.client.post('/api/close-position', json={'close_price': 62100})
        self.assertEqual(closed.status_code, 200)
        self.assertIsNone(closed.json()['active_position'])
        self.assertEqual(closed.json()['stats']['trades_closed'], 1)


if __name__ == '__main__':
    unittest.main()
