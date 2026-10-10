from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import lanzar_runpod as launcher
import runpod_entrypoint as entrypoint


class LauncherTests(unittest.TestCase):
    def run_launcher(self, *extra: str):
        output = io.StringIO()
        with redirect_stdout(output):
            result = launcher.main(["--image", "example/worker:v1", "--env-file",
                                    "nonexistent-test.env", *extra])
        return result, output.getvalue()

    @patch.dict(os.environ, {"RUNPOD_API_KEY": "runpod-secret",
                            "RADIO_API_TOKEN": "radio-secret", "HF_TOKEN": "hf-secret"}, clear=True)
    def test_preview_hides_secrets_and_never_calls_api(self):
        with patch.object(launcher, "urlopen") as request:
            result, output = self.run_launcher()
        self.assertEqual(result, 0)
        request.assert_not_called()
        for secret in ("runpod-secret", "radio-secret", "hf-secret"):
            self.assertNotIn(secret, output)

    @patch.dict(os.environ, {"RUNPOD_API_KEY": "runpod-secret",
                            "RADIO_API_TOKEN": "radio-secret", "WORKER_ID": "local",
                            "DEVICE": "cpu"}, clear=True)
    def test_create_transcription_pod_with_isolated_environment(self):
        response = io.BytesIO(b'{"id":"pod-test","env":{"token":"never-print"}}')
        with patch.object(launcher, "urlopen", return_value=response) as call:
            result, output = self.run_launcher("--create", "--capabilities", "transcription",
                                               "--network-volume-id", "volume-test",
                                               "--data-center-id", "dc-test")
        self.assertEqual(result, 0)
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.runpod.io/v2/pods")
        payload = json.loads(request.data)
        self.assertEqual(payload["mounts"]["network"][0]["path"], "/workspace")
        self.assertEqual(payload["env"], {"RADIO_API_TOKEN": "radio-secret",
                                         "WORKER_CAPABILITIES": "transcription"})
        self.assertNotIn("never-print", output)

    @patch.dict(os.environ, {"RUNPOD_API_KEY": "key", "RADIO_API_TOKEN": "token"}, clear=True)
    def test_uncertain_creation_is_not_retried(self):
        with patch.object(launcher, "urlopen", side_effect=URLError("timeout")) as call:
            result, _ = self.run_launcher("--create", "--capabilities", "transcription")
        self.assertEqual(result, 1)
        self.assertEqual(call.call_count, 1)

    @patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret", "RADIO_API_TOKEN": "radio-secret"}, clear=True)
    def test_forbidden_shows_reason_without_reflected_credentials(self):
        body = json.dumps({"detail": "Pods permission denied api-secret radio-secret"}).encode()
        error = HTTPError(launcher.API_URL, 403, "Forbidden", {}, io.BytesIO(body))
        output = io.StringIO()
        with patch.object(launcher, "urlopen", side_effect=error) as call, redirect_stderr(output):
            result, _ = self.run_launcher("--create", "--capabilities", "transcription")
        self.assertEqual(result, 1)
        self.assertIn("Pods permission denied", output.getvalue())
        self.assertNotIn("api-secret", output.getvalue())
        self.assertNotIn("radio-secret", output.getvalue())
        self.assertEqual(call.call_count, 1)

    def test_error_redacts_json_escaped_secret_and_handles_empty_body(self):
        secret = 'secret"value'
        body = json.dumps({"error": secret}).encode()
        error = HTTPError(launcher.API_URL, 403, "Forbidden", {}, io.BytesIO(body))
        self.assertNotIn("secret", launcher.api_error_detail(error, {"RUNPOD_API_KEY": secret}))
        empty = HTTPError(launcher.API_URL, 403, "Forbidden", {}, io.BytesIO(b""))
        self.assertIn("motivo", launcher.api_error_detail(empty, {}))

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_credentials_prevent_creation(self):
        with patch.object(launcher, "urlopen") as call, self.assertRaises(SystemExit):
            self.run_launcher("--create")
        call.assert_not_called()

    def test_network_volume_requires_location(self):
        with self.assertRaises(SystemExit):
            self.run_launcher("--network-volume-id", "volume-test")

    def test_container_executes_worker_with_arguments_and_volume_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            environment = {"RADIO_API_TOKEN": "token", "DEVICE": "cpu",
                           "MODELS_DIR": folder + "/models"}
            with patch.dict(os.environ, environment, clear=True), \
                 patch.object(sys, "argv", ["runpod_entrypoint.py", "--transcription-only", "--once"]), \
                 patch.object(entrypoint.os, "execv") as execute:
                entrypoint.main()
            from pathlib import Path
            self.assertTrue(Path(environment["MODELS_DIR"]).is_dir())
            self.assertEqual(execute.call_args.args[1][-2:], ["--transcription-only", "--once"])
            self.assertTrue(execute.call_args.args[1][2].endswith("worker_transcripcion.py"))

    @patch.dict(os.environ, {"RADIO_API_TOKEN": "token", "DEVICE": "cpu"}, clear=True)
    def test_diarization_requires_huggingface_token_before_start(self):
        with patch.object(sys, "argv", ["runpod_entrypoint.py"]), \
             patch.object(entrypoint.os, "execv") as execute, self.assertRaises(SystemExit):
            entrypoint.main()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
