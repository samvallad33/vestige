import json
from pathlib import Path
import unittest


FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "fixtures"
    / "sanhedrin-test-integrity-deltas"
)


class TestSanhedrinTestIntegrityDeltaFixtures(unittest.TestCase):
    def test_fixture_receipts_are_executable_contract_examples(self):
        fixtures = sorted(FIXTURE_DIR.glob("*.json"))
        self.assertEqual(
            [fixture.name for fixture in fixtures],
            [
                "justified-snapshot.json",
                "skipped-test.json",
                "unchanged-good.json",
                "weakened-assertion.json",
            ],
        )

        expected_decisions = {
            "justified-snapshot": "needs_human_review",
            "skipped-test": "downgraded",
            "unchanged-good": "accepted",
            "weakened-assertion": "downgraded",
        }

        for fixture in fixtures:
            with self.subTest(fixture=fixture.name):
                data = json.loads(fixture.read_text(encoding="utf-8"))
                receipt = data["receipt"]

                self.assertEqual(
                    receipt["schema"],
                    "vestige.sanhedrin.test_integrity_delta.v1",
                )
                self.assertEqual(data["expectedDecision"], receipt["decision"])
                self.assertEqual(expected_decisions[data["case"]], receipt["decision"])
                self.assertTrue(receipt["freshVerifier"]["checkedAfterLastRelevantEdit"])
                self.assertEqual(receipt["freshVerifier"]["exitCode"], 0)

                test_files = receipt["specSource"]["testFiles"]
                self.assertGreaterEqual(len(test_files), 1)
                for test_file in test_files:
                    self.assertTrue(test_file["path"])
                    self.assertRegex(
                        test_file["hashBeforeImplementation"],
                        r"^sha256:[0-9a-f]{64}$",
                    )
                    self.assertRegex(
                        test_file["hashAfterVerification"],
                        r"^sha256:[0-9a-f]{64}$",
                    )

    def test_downgrade_fixtures_have_mechanical_downgrade_evidence(self):
        for fixture in sorted(FIXTURE_DIR.glob("*.json")):
            data = json.loads(fixture.read_text(encoding="utf-8"))
            if data["expectedDecision"] != "downgraded":
                continue

            delta = data["receipt"]["delta"]
            has_downgrade_evidence = any(
                [
                    delta["removedOrDisabledTests"],
                    delta["removedAssertions"] > 0,
                    delta["weakenedExpectations"],
                    delta["snapshotChurnWithoutSourceChange"],
                    delta["coverageDelta"] < 0,
                    delta["mocksReplacingRealBoundary"],
                ]
            )
            self.assertTrue(has_downgrade_evidence, data["case"])


if __name__ == "__main__":
    unittest.main()
