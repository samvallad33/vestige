"""Native adapter boundary tests use injected MCP envelopes, no real kernel."""

import unittest

from kernel_adapter import TOOL_NAMES, KernelBoundaryError, NativeMemory


class FakeStack:
    def __init__(self, root, *, profile):
        self.catalog = [{"name": name} for name in TOOL_NAMES]
        self.calls = []
        self.aborted = False

    def start(self, session):
        self.session = session

    def tool(self, session, name, args):
        self.calls.append((name, args))
        if name == "smart_ingest":
            return {"structuredContent": {"nodeId": "mem-fixture"}}
        if name == "receipt":
            return {
                "structuredContent": {
                    "attestation": {
                        "verification": {
                            "chainValid": True,
                            "gateAllowed": True,
                            "locallyVerified": True,
                        }
                    }
                }
            }
        return {"structuredContent": {"results": []}}

    def abort(self):
        self.aborted = True

    def stop(self, session):
        pass

    def request(self, method):
        return {"chain": {"status": "ok", "breaks": 0}}

    def close(self):
        pass


class NativeAdapterTests(unittest.TestCase):
    def trial(self):
        return NativeMemory("/private/tmp/unused-kernel-fixture", factory=FakeStack)

    def test_gold_annotation_rejected_before_any_write(self):
        trial = self.trial()
        with self.assertRaises(KernelBoundaryError):
            trial.stage_record(
                {"content": "observation", "answer": "hidden"}, scope="case0", ordinal=0
            )
        self.assertEqual(trial.stack.calls, [])

    def test_history_is_explicit_write_with_verified_receipt(self):
        trial = self.trial()
        row = {
            "content": "Source says answer unknown; do not redact ordinary utterances."
        }
        result = trial.stage_record(row, scope="case0", ordinal=0)
        self.assertEqual(result["node_id"], "mem-fixture")
        self.assertEqual(
            [call[0] for call in trial.stack.calls], ["smart_ingest", "receipt"]
        )
        self.assertNotIn("derived_from", str(trial.stack.calls))
        trial.call("recall", {"handle": result["node_id"]})
        self.assertIsNone(trial.close()["model_calls"])

    def test_nested_annotations_are_not_committed(self):
        trial = self.trial()
        with self.assertRaises(KernelBoundaryError):
            trial.stage_record(
                {"turns": [{"content": "source", "has_answer": True}]},
                scope="case0",
                ordinal=0,
            )
        self.assertEqual(trial.stack.calls, [])

    def test_transport_failure_latches_before_next_operation(self):
        trial = self.trial()

        def broken(*args):
            raise OSError("fixture transport dropped")

        trial.stack.tool = broken
        with self.assertRaises(OSError):
            trial.call("recall", {"handle": "mem-fixture"})
        self.assertTrue(trial.stack.aborted)
        with self.assertRaises(KernelBoundaryError):
            trial.call("recall", {"handle": "mem-fixture"})

    def test_failed_attestation_latches_trial(self):
        trial = self.trial()
        trial.stack.tool = lambda *args: {
            "structuredContent": {"attestation": {"verification": {}}}
        }
        with self.assertRaises(KernelBoundaryError):
            trial.verify_receipt("mem-fixture")
        self.assertTrue(trial.failed)

    def test_malformed_and_logical_error_bodies_latch(self):
        for value in [
            None,
            [],
            "scalar",
            {},
            {"error": "fixture rejected"},
            {"success": False},
        ]:
            with self.subTest(value=value):
                trial = self.trial()
                trial.stack.tool = lambda *args, value=value: {
                    "structuredContent": value
                }
                with self.assertRaises(KernelBoundaryError):
                    trial.call("recall", {"handle": "mem-fixture"})
                self.assertTrue(trial.failed)
                with self.assertRaises(KernelBoundaryError):
                    trial.call("memory_status", {"view": "health"})

    def test_malformed_nested_attestation_latches(self):
        for attestation in [
            None,
            [],
            "scalar",
            {"verification": None},
            {"verification": []},
            {"verification": "scalar"},
        ]:
            with self.subTest(attestation=attestation):
                trial = self.trial()
                trial.stack.tool = lambda *args, value=attestation: {
                    "structuredContent": {"attestation": value}
                }
                with self.assertRaises(KernelBoundaryError):
                    trial.verify_receipt("mem-fixture")
                self.assertTrue(trial.failed)
                self.assertTrue(trial.stack.aborted)
                with self.assertRaises(KernelBoundaryError):
                    trial.call("memory_status", {"view": "health"})


if __name__ == "__main__":
    unittest.main()
