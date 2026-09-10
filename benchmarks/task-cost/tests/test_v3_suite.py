import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capture, compare, developer_suite, example, ledger, qualification, reconcile, trials


class V3Tests(unittest.TestCase):
    def fixture(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return example.create(Path(temp.name) / "bundle")

    def test_six_developer_evaluators_reject_bugs_and_accept_reference(self):
        result = developer_suite.self_test()
        self.assertEqual(len(result["cases"]), 6)
        self.assertEqual(result["model_runs"], 0)

    def test_wall_clock_and_frozen_schedule(self):
        root = self.fixture()
        with capture.measured_task(root, arm="native", case="cold-example", trial=0):
            pass
        report = ledger.evaluate(root)
        self.assertEqual(len(report["task_spans"]), 1)
        self.assertFalse(report["arms"]["native"]["task_timing_complete"])
        self.assertEqual(trials.schedule(root), trials.schedule(root))
        self.assertEqual(len(trials.schedule(root)), 4)
        with self.assertRaises(ValueError):
            with capture.measured_task(
                root, arm="native", case="cold-example", trial=0
            ):
                pass

    def test_reconciliation_detects_missing_and_mismatched_charges(self):
        root = self.fixture()
        report = ledger.evaluate(root)
        billing = {
            "currency": "USD",
            "source": "synthetic fixture",
            "charges": [
                {
                    "usage_format": e["usage_format"],
                    "request_id": e["provider_response_id"],
                    "usd": e["usd"],
                }
                for e in report["cost_events"]
                if e.get("provider_response_id")
            ],
        }
        path = root / "billing.json"
        path.write_text(json.dumps(billing))
        self.assertTrue(reconcile.reconcile(root, path)["matches_supplied_export"])
        billing["charges"][0]["usd"] = "1"
        path.write_text(json.dumps(billing))
        self.assertFalse(reconcile.reconcile(root, path)["matches_supplied_export"])

    def test_synthetic_never_qualifies_and_break_even_is_explicit_projection(self):
        root = self.fixture()
        self.assertEqual(
            qualification.qualify(root, "native", "candidate")["status"],
            "not_qualified",
        )
        self.assertEqual(
            qualification.projected_break_even("1", "0.2", "0.1")["tasks"], 10
        )
        self.assertIsNone(
            qualification.projected_break_even("1", "0.1", "0.2")["tasks"]
        )

    def test_empty_reconciliation_is_not_evidence(self):
        root = self.fixture()
        (root / "events.jsonl").write_text("")
        path = root / "billing.json"
        path.write_text(
            json.dumps({"currency": "USD", "source": "empty fixture", "charges": []})
        )
        result = reconcile.reconcile(root, path)
        self.assertFalse(result["matches_supplied_export"])
        self.assertEqual(result["matched_request_count"], 0)

    def test_driver_pipeline_and_duplicate_preflight_without_models(self):
        import zipfile

        root = self.fixture()
        contract = ledger.read_json(root / "contract.json")
        for name, body in {
            "driver.py": 'import json,pathlib,sys\ni=json.loads(pathlib.Path(sys.argv[1]).read_text())\n(pathlib.Path(i["checkout"])/"solution.py").write_text("answer=42\\n")\n',
            "evaluator.py": 'import pathlib,sys\nassert (pathlib.Path(sys.argv[1])/"solution.py").read_text()=="answer=42\\n"\n',
        }.items():
            (root / name).write_text(body)
        with zipfile.ZipFile(root / "source.zip", "w") as archive:
            archive.writestr("solution.py", "answer=0\n")

        def artifact(name):
            return {"path": name, "sha256": ledger.digest(root / name)}

        contract["driver"] = artifact("driver.py")
        for case in contract["cases"]:
            case["source"] = artifact("source.zip")
            case["evaluator"] = artifact("evaluator.py")
        (root / "contract.json").write_text(json.dumps(contract))
        (root / "contract.lock.json").unlink()
        ledger.freeze(root)
        (root / "events.jsonl").write_text("")
        selection = trials.schedule(root)[0]
        result = trials.run_trial(
            root, selection, trusted=True, output_parent=root.parent
        )
        self.assertEqual(result["status"], "success")
        report = ledger.evaluate(root)
        self.assertEqual(len(report["task_spans"]), 1)
        before = set(root.parent.iterdir())
        with self.assertRaises(ValueError):
            trials.run_trial(root, selection, trusted=True, output_parent=root.parent)
        self.assertEqual(set(root.parent.iterdir()), before)
        # A preexisting outcome alone also rejects before creating a checkout.
        lines = [
            json.loads(line)
            for line in (root / "events.jsonl").read_text().splitlines()
        ]
        (root / "events.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in lines if e["kind"] != "task_span")
        )
        with self.assertRaises(ValueError):
            trials.run_trial(root, selection, trusted=True, output_parent=root.parent)
        self.assertEqual(set(root.parent.iterdir()), before)

    def test_source_archive_rejects_traversal_and_symlinks(self):
        import zipfile

        root = self.fixture()
        checkout = root / "checkout"
        checkout.mkdir()
        for name, mode in [("../escaped.py", 0o100600), ("link", 0o120777)]:
            source = root / "bad.zip"
            with zipfile.ZipFile(source, "w") as archive:
                member = zipfile.ZipInfo(name)
                member.external_attr = mode << 16
                archive.writestr(member, "fixture")
            with self.assertRaises(ValueError):
                trials.unpack(source, checkout)
        self.assertEqual(list(checkout.iterdir()), [])

    def test_duplicate_provider_response_across_arms_is_rejected(self):
        root = self.fixture()
        events_path = root / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        native = next(
            e for e in events if e["kind"] == "request" and e["arm"] == "native"
        )
        candidate = next(
            e for e in events if e["kind"] == "request" and e["arm"] == "candidate"
        )
        candidate["response"] = native["response"]
        events_path.write_text("".join(json.dumps(e) + "\n" for e in events))
        with self.assertRaisesRegex(ValueError, "duplicate provider response id"):
            ledger.evaluate(root)

    def test_frozen_timing_requirement_prevents_complete_accounting(self):
        root = self.fixture()
        contract = ledger.read_json(root / "contract.json")
        contract["require_task_timing"] = True
        (root / "contract.json").write_text(json.dumps(contract))
        (root / "contract.lock.json").unlink()
        ledger.freeze(root)
        self.assertFalse(ledger.evaluate(root)["arms"]["native"]["accounting_complete"])
        for case in contract["cases"]:
            with capture.measured_task(root, arm="native", case=case["id"], trial=0):
                pass
        self.assertTrue(ledger.evaluate(root)["arms"]["native"]["accounting_complete"])

    def trial_fixture(self, driver, timeout=1):
        import zipfile

        root = self.fixture()
        contract = ledger.read_json(root / "contract.json")
        (root / "driver.py").write_text(driver)
        (root / "evaluator.py").write_text(
            'raise AssertionError("must not score failed driver")\n'
        )
        with zipfile.ZipFile(root / "source.zip", "w") as archive:
            archive.writestr("solution.py", "pass\n")

        def artifact(name):
            return {"path": name, "sha256": ledger.digest(root / name)}

        contract.update(driver=artifact("driver.py"), task_timeout_seconds=timeout)
        for case in contract["cases"]:
            case.update(
                source=artifact("source.zip"), evaluator=artifact("evaluator.py")
            )
        (root / "contract.json").write_text(json.dumps(contract))
        (root / "contract.lock.json").unlink()
        ledger.freeze(root)
        (root / "events.jsonl").write_text("")
        return root, trials.schedule(root)[0]

    def test_failed_and_timed_out_drivers_record_terminal_state_without_scoring(self):
        for program, expected in [
            ("raise SystemExit(3)", "failure"),
            ("import time; time.sleep(30)", "timeout"),
        ]:
            with self.subTest(expected=expected):
                root, selection = self.trial_fixture(program)
                result = trials.run_trial(
                    root,
                    selection,
                    trusted=True,
                    timeout_seconds=1,
                    output_parent=root.parent,
                )
                self.assertEqual(result["status"], expected)
                report = ledger.evaluate(root)
                self.assertEqual(
                    report["arms"][selection["arm"]]["outcome_counts"][expected], 1
                )
                self.assertEqual(
                    report["task_spans"][0]["status"],
                    "timeout" if expected == "timeout" else "failed",
                )
                self.assertEqual(list(root.glob("evaluation-*.json")), [])
                self.assertFalse(
                    report["arms"][selection["arm"]]["accounting_complete"]
                )

    def test_driver_and_timeout_tampering_rejected_before_execution(self):
        root, selection = self.trial_fixture("raise SystemExit(0)")
        before = set(root.parent.iterdir())
        with self.assertRaises(ValueError):
            trials.run_trial(
                root,
                selection,
                trusted=True,
                timeout_seconds=2,
                output_parent=root.parent,
            )
        (root / "driver.py").write_text("raise SystemExit(9)")
        with self.assertRaises(ValueError):
            trials.run_trial(
                root,
                selection,
                trusted=True,
                timeout_seconds=1,
                output_parent=root.parent,
            )
        self.assertEqual(set(root.parent.iterdir()), before)

    def test_reconciliation_rejects_duplicate_and_nonfinite_charges(self):
        root = self.fixture()
        path = root / "bad-billing.json"
        entry = {
            "usage_format": "openai_responses",
            "request_id": "fixture",
            "usd": "0.1",
        }
        for charges in [
            [entry, entry],
            [dict(entry, usd="NaN")],
            [dict(entry, usd="-1")],
        ]:
            path.write_text(
                json.dumps({"currency": "USD", "source": "fixture", "charges": charges})
            )
            with self.assertRaises(ValueError):
                reconcile.reconcile(root, path)
