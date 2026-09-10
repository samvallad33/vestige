import copy
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example import create
from ledger import amount, artifact, digest, evaluate, freeze, normalize, price_usage, record


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = create(Path(self.temp.name) / "bundle")

    def events(self):
        return [json.loads(line) for line in (self.root / "events.jsonl").read_text().splitlines()]

    def rewrite(self, events):
        (self.root / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))

    def change_response(self, events, index, callback):
        path = self.root / events[index]["response"]["path"]
        value = json.loads(path.read_text())
        callback(value)
        path.write_text(json.dumps(value))
        events[index]["response"]["sha256"] = digest(path)
        self.rewrite(events)

    def test_counts_failed_tasks_and_overhead_in_success_cost(self):
        report = evaluate(self.root)
        self.assertEqual(report["evidence_kind"], "synthetic")
        for arm in report["arms"].values():
            self.assertTrue(arm["accounting_complete"])
            self.assertEqual(arm["success_rate"], 0.5)
            self.assertEqual(Decimal(arm["total_usd"]), Decimal("0.01488"))
            self.assertEqual(arm["total_usd"], arm["usd_per_success"])
            self.assertEqual(arm["known_token_units"]["output"], 200)

    def test_openai_reasoning_and_cache_not_double_counted(self):
        value = normalize("openai_responses", {"input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 800}, "output_tokens": 100,
            "output_tokens_details": {"reasoning_tokens": 90}})
        cost = price_usage(value, {"usd_per_million_tokens": {
            "uncached_input": "2", "cached_input": "0.2", "output": "8"}})
        self.assertEqual(cost, Decimal("0.00136"))
        self.assertEqual(value["reasoning_tokens_subset_of_output"], 90)

    def test_anthropic_cache_ttls_are_separate_units(self):
        value = normalize("anthropic_messages", {"input_tokens": 10, "output_tokens": 5,
            "cache_read_input_tokens": 30, "cache_creation_input_tokens": 50,
            "cache_creation": {"ephemeral_5m_input_tokens": 20, "ephemeral_1h_input_tokens": 30}})
        self.assertEqual(value["tokens"], {"uncached_input": 10, "output": 5,
            "cached_input": 30, "cache_write_5m": 20, "cache_write_1h": 30})

    def test_missing_cache_ttl_is_unknown(self):
        self.assertIsNone(normalize("anthropic_messages", {"input_tokens": 10, "output_tokens": 5,
            "cache_read_input_tokens": 30, "cache_creation_input_tokens": 50}))

    def test_missing_openai_cache_split_is_unknown(self):
        self.assertIsNone(normalize("openai_responses", {"input_tokens": 10, "output_tokens": 5}))

    def test_provider_total_must_reconcile(self):
        with self.assertRaisesRegex(ValueError, "total_tokens differs"):
            normalize("openai_responses", {"input_tokens": 10, "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 0}, "total_tokens": 20})

    def test_missing_usage_blocks_complete_total(self):
        events = self.events()
        self.change_response(events, 0, lambda response: response.pop("usage"))
        arm = evaluate(self.root)["arms"]["native"]
        self.assertFalse(arm["accounting_complete"])
        self.assertIsNone(arm["total_usd"])
        self.assertEqual(len(arm["unknown_cost_events"]), 1)

    def test_missing_price_is_not_zero(self):
        normalized = normalize("openai_responses", {"input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1})
        self.assertIsNone(price_usage(normalized, {"usd_per_million_tokens": {"output": "1"}}))

    def test_zero_successes_is_undefined(self):
        events = self.events()
        for event in events:
            if event["kind"] == "outcome":
                event["status"] = "failure"
        self.rewrite(events)
        self.assertIsNone(evaluate(self.root)["arms"]["native"]["usd_per_success"])

    def test_missing_planned_outcome_cannot_disappear(self):
        self.rewrite(self.events()[0:1] + self.events()[2:])
        arm = evaluate(self.root)["arms"]["native"]
        self.assertFalse(arm["accounting_complete"])
        self.assertEqual(len(arm["missing_outcomes"]), 1)

    def test_retry_is_counted_but_duplicate_export_rejected(self):
        events = self.events()
        retry = copy.deepcopy(events[0])
        retry["id"] = "retry"
        response = json.loads((self.root / retry["response"]["path"]).read_text())
        response["id"] = "provider-retry"
        path = self.root / "retry-response.json"
        path.write_text(json.dumps(response))
        retry["response"] = {"path": path.name, "sha256": digest(path)}
        record(self.root, retry)
        self.assertEqual(evaluate(self.root)["arms"]["native"]["request_count"], 3)
        retry["id"] = "same-response-new-event"
        record(self.root, retry)
        with self.assertRaisesRegex(ValueError, "duplicate provider response"):
            evaluate(self.root)

    def test_unknown_overhead_prevents_complete_accounting(self):
        self.rewrite([event for event in self.events() if event["kind"] != "overhead_coverage"])
        self.assertFalse(evaluate(self.root)["arms"]["native"]["accounting_complete"])

    def test_unknown_overhead_amount_is_not_zero(self):
        events = self.events()
        for event in events:
            if event["kind"] == "overhead":
                event["usd"] = None
        self.rewrite(events)
        self.assertIsNone(evaluate(self.root)["arms"]["native"]["total_usd"])

    def test_embedding_request_does_not_stand_in_for_agent_run(self):
        events = self.events()
        for event in events:
            if event["kind"] == "request":
                event["phase"] = "embedding"
        self.rewrite(events)
        self.assertEqual(len(evaluate(self.root)["arms"]["native"]["tasks_without_requests"]), 2)

    def test_artifact_tampering_rejected(self):
        (self.root / "stimulus.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            evaluate(self.root)

    def test_contract_tampering_rejected(self):
        with (self.root / "contract.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "changed after freeze"):
            evaluate(self.root)

    def test_refreeze_does_not_overwrite(self):
        with self.assertRaises(FileExistsError):
            freeze(self.root)

    def test_model_parity_checked_before_freeze(self):
        path = self.root / "contract.json"
        contract = json.loads(path.read_text())
        contract["arms"]["candidate"]["effort"] = "different"
        path.write_text(json.dumps(contract))
        (self.root / "contract.lock.json").unlink()
        with self.assertRaisesRegex(ValueError, "parity mismatch"):
            freeze(self.root)

    def test_accountant_identity_checked(self):
        path = self.root / "contract.lock.json"
        lock = json.loads(path.read_text())
        lock["accountant_sha256"] = "0" * 64
        path.write_text(json.dumps(lock))
        with self.assertRaisesRegex(ValueError, "accountant changed"):
            evaluate(self.root)

    def test_negative_boolean_or_impossible_usage_rejected(self):
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize("openai_responses", {"input_tokens": invalid})
        with self.assertRaisesRegex(ValueError, "exceeds total"):
            normalize("openai_responses", {"input_tokens": 1, "output_tokens": 1,
                "input_tokens_details": {"cached_tokens": 2}})

    def test_nonfinite_or_float_prices_rejected(self):
        for value in ("NaN", "Infinity", "-1", 1.2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                amount(value)

    def test_path_escape_rejected(self):
        outside = Path(self.temp.name) / "outside.json"
        outside.write_text("{}")
        with self.assertRaisesRegex(ValueError, "escapes bundle"):
            artifact(self.root, {"path": "../outside.json", "sha256": digest(outside)})

    def test_unknown_arm_and_duplicate_event_rejected(self):
        events = self.events()
        self.rewrite(events + [events[0]])
        with self.assertRaisesRegex(ValueError, "duplicate event"):
            evaluate(self.root)
        events[0]["arm"] = "invented"
        self.rewrite(events)
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            evaluate(self.root)

    def test_response_model_must_match_price(self):
        self.change_response(self.events(), 0, lambda response: response.update(model="other-model"))
        with self.assertRaisesRegex(ValueError, "model mismatch"):
            evaluate(self.root)


if __name__ == "__main__":
    unittest.main()
