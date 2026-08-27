from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worker_transcripcion import nonnegative_config_integer, run_local_pipeline  # noqa: E402


class WorkerConfigTests(unittest.TestCase):
    def test_batch_size_defaults_to_sequential(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(nonnegative_config_integer({}, "WHISPER_BATCH_SIZE", 0), 0)

    def test_batch_size_accepts_explicit_batch_profile(self) -> None:
        with patch.dict(os.environ, {"WHISPER_BATCH_SIZE": "12"}, clear=True):
            self.assertEqual(nonnegative_config_integer({}, "WHISPER_BATCH_SIZE", 0), 12)

    def test_batch_size_rejects_negative_or_non_integer_values(self) -> None:
        for raw in ("-1", "doce"):
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {"WHISPER_BATCH_SIZE": raw}, clear=True):
                    with self.assertRaises(SystemExit):
                        nonnegative_config_integer({}, "WHISPER_BATCH_SIZE", 0)

    @patch("worker_transcripcion.subprocess.run")
    def test_subprocess_pipeline_receives_configured_batch_size(self, mocked_run) -> None:
        mocked_run.return_value = MagicMock(returncode=0)
        cfg = SimpleNamespace(
            language="es",
            device="cuda",
            transcription_engine="faster-whisper",
            whisper_model="large-v3-local",
            pyannote_model="community-1",
            diarization_engine="pyannote",
            models_dir=PROJECT_ROOT / "modelos",
            whisper_batch_size=0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_local_pipeline(cfg, root / "audio.mp3", root / "out")

        command = mocked_run.call_args.args[0]
        index = command.index("--batch-size")
        self.assertEqual(command[index + 1], "0")


if __name__ == "__main__":
    unittest.main()
