import unittest
from unittest.mock import MagicMock, patch

import requests

import worker_transcripcion as worker


def response(status: int) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp.url = "https://example.test/complete"
    return resp


class PostWithRetryTest(unittest.TestCase):
    def run_post(self, outcomes: list, delays=(2, 5)):
        session = MagicMock()
        session.post.side_effect = outcomes
        with patch.object(worker.time, "sleep") as sleep:
            try:
                result = worker.post_with_retry(session, "https://example.test/complete", delays, "job")
            except Exception as exc:
                result = exc
        return result, session.post.call_count, [call.args[0] for call in sleep.call_args_list]

    def test_reintenta_corte_ssl_y_luego_tiene_exito(self):
        result, calls, sleeps = self.run_post([requests.exceptions.SSLError("EOF"), response(200)])
        self.assertEqual(result.status_code, 200)
        self.assertEqual(calls, 2)
        self.assertEqual(sleeps, [2])

    def test_reintenta_5xx_y_lectura_truncada(self):
        result, calls, sleeps = self.run_post(
            [response(502), requests.exceptions.ChunkedEncodingError("IncompleteRead"), response(200)])
        self.assertEqual(result.status_code, 200)
        self.assertEqual(sleeps, [2, 5])

    def test_no_reintenta_4xx(self):
        result, calls, sleeps = self.run_post([response(409)])
        self.assertIsInstance(result, requests.HTTPError)
        self.assertEqual(calls, 1)
        self.assertEqual(sleeps, [])

    def test_se_rinde_tras_agotar_esperas(self):
        result, calls, _ = self.run_post([requests.ConnectionError("x")] * 3)
        self.assertIsInstance(result, requests.ConnectionError)
        self.assertEqual(calls, 3)

    def test_ultimo_intento_con_5xx_lanza_http_error(self):
        result, calls, _ = self.run_post([response(503)] * 3)
        self.assertIsInstance(result, requests.HTTPError)
        self.assertEqual(calls, 3)


class HeartbeatTest(unittest.TestCase):
    def test_tras_un_fallo_reintenta_rapido(self):
        thread = worker.HeartbeatThread(MagicMock(), MagicMock(), "job", "lease", interval=30)
        waits = []

        def fake_wait(timeout):
            waits.append(timeout)
            return len(waits) > 3

        thread._stop.wait = fake_wait
        with patch.object(worker, "send_heartbeat", side_effect=[requests.ConnectionError("x"), None, None]):
            thread.run()
        self.assertEqual(waits, [30, 5.0, 30, 30])


if __name__ == "__main__":
    unittest.main()
