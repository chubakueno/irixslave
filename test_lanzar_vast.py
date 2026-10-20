from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import lanzar_vast as launcher

# La API ordena por precio; el lanzador debe anteponer la GPU preferida (4070S).
OFFERS = {"offers": [
    {"id": 100, "gpu_name": "RTX 4070", "num_gpus": 1, "dph_total": 0.15, "cuda_max_good": 13.0},
    {"id": 101, "gpu_name": "RTX 4070S", "num_gpus": 1, "dph_total": 0.20, "cuda_max_good": 13.0},
    {"id": 102, "gpu_name": "RTX 4070S", "num_gpus": 1, "dph_total": 0.25, "cuda_max_good": 13.0},
]}


def response(body: dict) -> io.BytesIO:
    return io.BytesIO(json.dumps(body).encode())


def rejection(code: int, body: dict) -> HTTPError:
    return HTTPError(launcher.API_BASE, code, "error", {}, response(body))


class VastLauncherTests(unittest.TestCase):
    def test_gpu_list_keeps_preference_and_drops_unrequested_models(self):
        gpus = launcher.gpu_preference("RTX_4070S, RTX 4070")
        self.assertEqual(gpus, ["RTX 4070S", "RTX 4070"])
        offers = OFFERS["offers"] + [{"id": 103, "gpu_name": "RTX 3060", "dph_total": 0.05}]
        self.assertEqual([o["id"] for o in launcher.rank_offers(offers, gpus)], [101, 102, 100])

    def run_launcher(self, *extra: str):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            result = launcher.main(["--image", "example/worker:v1", "--env-file",
                                    "nonexistent-test.env", *extra])
        return result, output.getvalue(), errors.getvalue()

    @patch.dict(os.environ, {"RADIO_API_TOKEN": "radio-secret", "HF_TOKEN": "hf-secret"}, clear=True)
    def test_preview_without_key_searches_anonymously_and_hides_secrets(self):
        with patch.object(launcher, "urlopen", return_value=response(OFFERS)) as call:
            result, output, _ = self.run_launcher()
        self.assertEqual(result, 0)
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("Authorization", call.call_args.args[0].headers)
        self.assertNotIn("radio-secret", output)
        self.assertNotIn("hf-secret", output)

    @patch.dict(os.environ, {"VAST_API_KEY": "vast-secret", "RADIO_API_TOKEN": "radio-secret"}, clear=True)
    def test_preview_only_searches_offers(self):
        with patch.object(launcher, "urlopen", return_value=response(OFFERS)) as call:
            result, output, _ = self.run_launcher("--max-price", "0.4")
        self.assertEqual(result, 0)
        self.assertEqual(call.call_count, 1)
        request = call.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        query = json.loads(request.data)
        self.assertEqual(query["gpu_name"], {"in": ["RTX 4070S", "RTX 4070S Ti", "RTX 4070 Ti", "RTX 4070"]})
        self.assertEqual(query["cuda_max_good"], {"gte": 12.8})
        self.assertEqual(query["dph_total"], {"lte": 0.4})
        self.assertLess(output.index("oferta 101"), output.index("oferta 100"))
        self.assertNotIn("vast-secret", output)

    @patch.dict(os.environ, {"VAST_API_KEY": "key", "RADIO_API_TOKEN": "radio-secret",
                            "WORKER_ID": "local", "REGISTRY_USERNAME": "user",
                            "REGISTRY_PASSWORD": "dckr_pat_x"}, clear=True)
    def test_create_rents_cheapest_offer_with_worker_entrypoint(self):
        replies = [response(OFFERS), response({"success": True, "new_contract": 555})]
        with patch.object(launcher, "urlopen", side_effect=replies) as call:
            result, output, _ = self.run_launcher("--create", "--capabilities", "transcription")
        self.assertEqual(result, 0)
        request = call.call_args.args[0]
        self.assertEqual(request.get_method(), "PUT")
        self.assertTrue(request.full_url.endswith("/asks/101/"))
        body = json.loads(request.data)
        self.assertEqual(body["runtype"], "args")
        self.assertEqual(body["args"], ["--live", "--persistent-models"])
        self.assertEqual(body["env"], {"RADIO_API_TOKEN": "radio-secret",
                                      "WORKER_CAPABILITIES": "transcription"})
        self.assertEqual(body["image_login"], "-u user -p dckr_pat_x docker.io")
        self.assertIn("555", output)
        self.assertNotIn("dckr_pat_x", output)

    @patch.dict(os.environ, {"VAST_API_KEY": "key", "RADIO_API_TOKEN": "token"}, clear=True)
    def test_taken_offer_falls_back_to_next(self):
        replies = [response(OFFERS), rejection(410, {"error": "no_such_ask"}),
                   response({"success": True, "new_contract": 556})]
        with patch.object(launcher, "urlopen", side_effect=replies) as call:
            result, output, _ = self.run_launcher("--create", "--capabilities", "transcription")
        self.assertEqual(result, 0)
        self.assertTrue(call.call_args.args[0].full_url.endswith("/asks/102/"))
        self.assertIn("556", output)

    @patch.dict(os.environ, {"VAST_API_KEY": "key", "RADIO_API_TOKEN": "token"}, clear=True)
    def test_uncertain_rental_is_not_retried(self):
        replies = [response(OFFERS), URLError("timeout")]
        with patch.object(launcher, "urlopen", side_effect=replies) as call:
            result, _, errors = self.run_launcher("--create", "--capabilities", "transcription")
        self.assertEqual(result, 1)
        self.assertEqual(call.call_count, 2)
        self.assertIn("antes de repetir", errors)

    @patch.dict(os.environ, {"VAST_API_KEY": "api-secret", "RADIO_API_TOKEN": "radio-secret"}, clear=True)
    def test_forbidden_stops_without_leaking_credentials(self):
        replies = [response(OFFERS), rejection(403, {"msg": "denied api-secret radio-secret"})]
        with patch.object(launcher, "urlopen", side_effect=replies) as call:
            result, _, errors = self.run_launcher("--create", "--capabilities", "transcription")
        self.assertEqual(result, 1)
        self.assertEqual(call.call_count, 2)
        self.assertIn("denied", errors)
        self.assertNotIn("api-secret", errors)
        self.assertNotIn("radio-secret", errors)

    @patch.dict(os.environ, {"VAST_API_KEY": "key", "RADIO_API_TOKEN": "token"}, clear=True)
    def test_explicit_offer_skips_search(self):
        with patch.object(launcher, "urlopen",
                          return_value=response({"success": True, "new_contract": 7})) as call:
            result, _, _ = self.run_launcher("--create", "--capabilities", "transcription",
                                             "--offer-id", "999")
        self.assertEqual(result, 0)
        self.assertEqual(call.call_count, 1)
        self.assertTrue(call.call_args.args[0].full_url.endswith("/asks/999/"))

    @patch.dict(os.environ, {"VAST_API_KEY": "key", "RADIO_API_TOKEN": "token"}, clear=True)
    def test_missing_huggingface_token_prevents_rental(self):
        with patch.object(launcher, "urlopen") as call, self.assertRaises(SystemExit):
            self.run_launcher("--create")
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
