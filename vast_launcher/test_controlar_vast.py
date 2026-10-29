import unittest
from unittest.mock import patch
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import controlar_vast as controller

CAPS = ["transcription", "diarization"]
EMPTY = {"capabilities": CAPS, "available_jobs": 0, "active_leases": 0, "next_available_at": None}


class ControllerTests(unittest.TestCase):
    def test_queue_requires_no_jobs_leases_or_retries(self):
        self.assertTrue(controller.queue_empty(EMPTY, CAPS))
        for change in ({"available_jobs": 1}, {"active_leases": 1},
                       {"next_available_at": "2026-10-29T15:00:00Z"}):
            self.assertFalse(controller.queue_empty({**EMPTY, **change}, CAPS))

    def test_invalid_response_cannot_trigger_deletion(self):
        for change in ({"available_jobs": False}, {"active_leases": -1},
                       {"available_jobs": "0"}, {"capabilities": ["transcription"]}):
            with self.assertRaises(ValueError):
                controller.queue_empty({**EMPTY, **change}, CAPS)
        with self.assertRaises(ValueError):
            controller.queue_empty({k: v for k, v in EMPTY.items() if k != "next_available_at"}, CAPS)

    def test_continuous_empty_window_and_reset_on_jobs_or_error(self):
        w = controller.IdleWindow(600, 90)
        for now in range(0, 600, 30):
            self.assertFalse(w.observe(True, now))
        self.assertTrue(w.observe(True, 600))
        self.assertFalse(w.observe(False, 630))
        self.assertFalse(w.observe(True, 660))
        w.reset()  # network/schema failure
        self.assertFalse(w.observe(True, 690))
        self.assertEqual(w.since, 690)

    def test_suspend_or_clock_jump_restarts_window(self):
        for later in (900, -1):
            w = controller.IdleWindow(600, 90)
            w.observe(True, 0)
            self.assertFalse(w.observe(True, later))
            self.assertEqual(w.since, later)

    def test_identity_requires_all_local_constraints(self):
        a = Namespace(instance=123, expect_machine=456, expect_label="worker", expect_image="image")
        valid = dict(id=123, machine_id=456, label="worker", image_uuid="image")
        self.assertTrue(controller.identity(valid, a))
        for key in valid:
            self.assertFalse(controller.identity({**valid, key: "other"}, a))

    def test_delete_only_after_window_and_never_retry_uncertain_delete(self):
        instance = {"instances": dict(id=123, machine_id=456, label="worker",
                                      image_uuid="image", actual_status="running")}
        calls = []
        def api(method, url, token, worker_id=None):
            calls.append((method, url, token))
            if method == "DELETE":
                raise TimeoutError("uncertain")
            return EMPTY if "/jobs/status" in url else instance
        with TemporaryDirectory() as temp, patch.object(controller, "read_env", return_value={
            "RADIO_API_TOKEN": "radio", "VAST_API_KEY": "vast"
        }), patch.object(controller, "api", side_effect=api), \
                patch.object(controller.time, "time", side_effect=[0, 0, 30, 30]), \
                patch.object(controller.time, "sleep"):
            rc = controller.main(["--instance", "123", "--expect-machine", "456",
                                  "--expect-label", "worker", "--expect-image", "image",
                                  "--env-file", str(Path(temp) / ".env"),
                                  "--idle-seconds", "30", "--destroy"])
            self.assertTrue((Path(temp) / ".vast-controller/123.delete-attempted").exists())
            before = len(calls)
            controller.main(["--instance", "123", "--expect-machine", "456",
                             "--env-file", str(Path(temp) / ".env"), "--destroy"])
            self.assertEqual(len(calls), before)
        self.assertEqual(rc, 0)
        self.assertEqual(sum(method == "DELETE" for method, _, _ in calls), 1)
        for _, url, token in calls:
            self.assertEqual(token, "radio" if "/jobs/status" in url else "vast")


if __name__ == "__main__":
    unittest.main()
