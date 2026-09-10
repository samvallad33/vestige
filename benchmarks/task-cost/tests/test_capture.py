from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture import measured_call
from example import create
from ledger import evaluate


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = create(Path(self.temp.name) / "bundle")
        self.metadata = {"id": "captured-call", "kind": "request", "arm": "native",
                         "case": "cold-example", "trial": 0, "phase": "agent",
                         "format": "openai_responses", "price_id": "fixture"}
        self.request = {"model": "fixture-model-v1", "input": "Synthetic request"}

    def test_captures_sdk_model_dump_and_returns_original_object(self):
        class Response:
            def model_dump(self, mode):
                return {"id": "sdk-response", "model": "fixture-model-v1", "usage": {
                    "input_tokens": 10, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 2}}
        response = Response()
        result = measured_call(self.root, self.metadata, self.request, lambda **kwargs: response)
        self.assertIs(result, response)
        report = evaluate(self.root)
        self.assertEqual(report["arms"]["native"]["request_count"], 3)
        self.assertTrue(report["arms"]["native"]["accounting_complete"])
        self.assertIsInstance(report["cost_events"][-1]["elapsed_ms"], int)

    def test_failure_preserves_exception_and_unknown_charge_without_secret_message(self):
        def fail(**kwargs):
            raise TimeoutError("PRIVATE-EXCEPTION-CONTENT")
        with self.assertRaises(TimeoutError):
            measured_call(self.root, self.metadata, self.request, fail)
        arm = evaluate(self.root)["arms"]["native"]
        self.assertIsNone(arm["total_usd"])
        for path in (self.root / "captures").glob("*.json"):
            self.assertNotIn("PRIVATE-EXCEPTION-CONTENT", path.read_text())

    def test_request_is_saved_before_call(self):
        def inspect(**kwargs):
            paths = list((self.root / "captures").glob("*-request.json"))
            self.assertEqual(len(paths), 1)
            self.assertEqual(json.loads(paths[0].read_text()), self.request)
            return {"usage": None}
        measured_call(self.root, self.metadata, self.request, inspect)

    def test_streaming_and_credentials_rejected_before_call(self):
        for addition in ({"stream": True}, {"Authorization": "fixture-secret"},
                         {"options": {"api_key": "fixture-secret"}}):
            called = []
            with self.subTest(addition=addition), self.assertRaises(ValueError):
                measured_call(self.root, self.metadata, {**self.request, **addition},
                              lambda **kwargs: called.append(True))
            self.assertFalse(called)

    def test_frozen_model_enforced_before_call(self):
        with self.assertRaisesRegex(ValueError, "model must match"):
            measured_call(self.root, self.metadata, {"model": "different"}, lambda **kwargs: {})

    def test_unserializable_response_still_records_unknown_billing(self):
        with self.assertRaises(ValueError):
            measured_call(self.root, self.metadata, self.request, lambda **kwargs: object())
        self.assertIsNone(evaluate(self.root)["arms"]["native"]["total_usd"])

    def test_duplicate_event_rejected_before_second_call(self):
        measured_call(self.root, self.metadata, self.request, lambda **kwargs: {"usage": None})
        called = []
        with self.assertRaisesRegex(ValueError, "duplicate event"):
            measured_call(self.root, self.metadata, self.request, lambda **kwargs: called.append(True))
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
