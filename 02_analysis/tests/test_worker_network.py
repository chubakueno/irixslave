from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worker_transcripcion import (  # noqa: E402
    PreparedJob,
    download_audio,
    post_with_retry,
    process_prepared_job,
)


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}", response=self
            )


class FakeSession:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def post(self, *args, **kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeDownloadResponse(FakeResponse):
    headers = {"Content-Type": "audio/mpeg"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def iter_content(self, chunk_size: int):
        yield b"audio"


class FakeDownloadSession:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def get(self, *args, **kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class WorkerNetworkTests(unittest.TestCase):
    def call(self, session: FakeSession, attempts: int = 4) -> None:
        post_with_retry(
            session,  # type: ignore[arg-type]
            "https://example.test/complete",
            headers={"Authorization": "redacted"},
            json_payload={"text": "ok"},
            timeout=1,
            max_attempts=attempts,
            operation="test complete",
        )

    @patch("worker_transcripcion.time.sleep", return_value=None)
    def test_retries_connection_error_then_succeeds(self, mocked_sleep) -> None:
        session = FakeSession(
            [requests.ConnectionError("reset"), FakeResponse(200)]
        )
        self.call(session)
        self.assertEqual(session.calls, 2)
        mocked_sleep.assert_called_once()

    @patch("worker_transcripcion.time.sleep", return_value=None)
    def test_retries_retryable_http_status(self, mocked_sleep) -> None:
        session = FakeSession([FakeResponse(503), FakeResponse(200)])
        self.call(session)
        self.assertEqual(session.calls, 2)
        mocked_sleep.assert_called_once()

    @patch("worker_transcripcion.time.sleep", return_value=None)
    def test_does_not_retry_conflict(self, mocked_sleep) -> None:
        session = FakeSession([FakeResponse(409), FakeResponse(200)])
        with self.assertRaises(requests.HTTPError):
            self.call(session)
        self.assertEqual(session.calls, 1)
        mocked_sleep.assert_not_called()

    @patch("worker_transcripcion.time.sleep", return_value=None)
    def test_download_retries_ssl_transport_error(self, mocked_sleep) -> None:
        session = FakeDownloadSession(
            [requests.exceptions.SSLError("eof"), FakeDownloadResponse(200)]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_audio(
                session,  # type: ignore[arg-type]
                "https://example.test/audio.mp3?signature=redacted",
                {},
                Path(temp_dir),
            )
            self.assertEqual(result.read_bytes(), b"audio")
        self.assertEqual(session.calls, 2)
        mocked_sleep.assert_called_once()

    @patch("worker_transcripcion.fail_job")
    @patch("worker_transcripcion.complete_job", side_effect=requests.ConnectionError("reset"))
    def test_complete_error_preserves_result_without_reporting_fail(
        self, mocked_complete, mocked_fail
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio.mp3"
            audio.write_bytes(b"audio")
            session = MagicMock()
            heartbeat = MagicMock()
            temporary = MagicMock(name="temporary_directory")
            temporary.name = str(root / "job")
            Path(temporary.name).mkdir()
            prepared = PreparedJob(
                job={},
                job_id="job-1",
                lease_id="lease-1",
                session=session,
                tmp_dir=temporary,
                audio_path=audio,
                heartbeat=heartbeat,
            )
            cfg = SimpleNamespace(
                language="es",
                results_dir=root / "results",
                transcription_engine="faster-whisper",
                whisper_model="large-v3-local",
                diarization_engine="pyannote",
                pyannote_model="community-1",
            )
            engine = MagicMock()
            engine.process.return_value = (
                {
                    "turns": [{"speaker": "SPEAKER_00", "text": "hola"}],
                    "units": [
                        {
                            "text": "hola",
                            "start": 0.0,
                            "end": 0.5,
                            "speaker": "SPEAKER_00",
                            "probability": 0.9,
                        }
                    ],
                },
                {"language": "es"},
            )

            with self.assertRaises(requests.ConnectionError):
                process_prepared_job(cfg, prepared, live=True, engine=engine)

            self.assertTrue((cfg.results_dir / "job-1.json").is_file())
            mocked_complete.assert_called_once()
            mocked_fail.assert_not_called()
            heartbeat.stop.assert_called_once()
            temporary.cleanup.assert_called_once()
            session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
