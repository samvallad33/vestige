import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


BENCHMARK = Path(__file__).resolve().parents[1]
EVIDENCE = BENCHMARK / "evidence"
sys.path.insert(0, str(BENCHMARK))

import recheck_application  # noqa: E402


evaluator = recheck_application.evaluator


def _flags(report):
    return [(item["name"], item["pass"]) for item in report["checks"]]


def _copy_reference(destination):
    shutil.copytree(evaluator.FIXTURE, destination)
    evaluator._replace(
        destination / "runtime.conf",
        "route.RECONCILE=CEDAR@7",
        "route.RECONCILE=CYPRESS@7",
    )
    evaluator._write_decision(destination, ["runtime.conf"])


def _copy_recorded_contract(destination):
    destination.mkdir()
    shutil.copytree(EVIDENCE / "inputs", destination / "inputs")
    for arm in recheck_application.ARMS:
        arm_root = destination / arm
        arm_root.mkdir()
        shutil.copytree(EVIDENCE / arm / "project", arm_root / "project")


class RecheckApplicationTests(unittest.TestCase):
    def test_recorded_projects_reproduce_original_application_results(self):
        with tempfile.TemporaryDirectory(prefix="application-recheck-test-") as temporary:
            output = Path(temporary) / "output"
            result = recheck_application.run_recheck(EVIDENCE, output)

            self.assertTrue(result["all_application_checks_pass"], result["arms"])
            self.assertFalse(result["model_execution_performed"])
            self.assertEqual(result["recheck_kind"], "recorded-final-application-only")
            self.assertEqual([arm["arm"] for arm in result["arms"]], list(recheck_application.ARMS))
            for arm in result["arms"]:
                self.assertTrue(arm["copy_fidelity"], arm)
                self.assertTrue(arm["application_files_unchanged_by_adaptation"], arm)
                self.assertTrue(arm["recorded_protected_inputs_match"], arm)
                self.assertTrue(arm["disposable_input_unchanged_after_adaptation"], arm)
                self.assertTrue(arm["recorded_evidence_unchanged"], arm)
                self.assertEqual(
                    [item["path"] for item in arm["controlled_disposable_adaptations"]],
                    ["reproduce.sh"],
                )
                adaptation = arm["controlled_disposable_adaptations"][0]
                self.assertNotEqual(
                    adaptation["recorded"]["sha256"], adaptation["portable"]["sha256"]
                )
                self.assertEqual((arm["score"], arm["total"]), (21, 21))
                current = json.loads((output / arm["arm"] / "report.json").read_text())
                recorded = json.loads(
                    (EVIDENCE / arm["arm"] / "evaluation.json").read_text()
                )
                self.assertEqual(_flags(current), _flags(recorded), arm["arm"])

            historical = result["historical_model_run"]["mcp-memory-service"]
            self.assertTrue(historical["timeout"])
            self.assertEqual(historical["timeout_seconds"], 900)
            self.assertFalse(historical["model_finished_naturally"])
            self.assertTrue(historical["final_application_files_passed"])
            self.assertEqual(
                json.loads((output / recheck_application.RESULT_FILE).read_text()), result
            )

    def test_public_baseline_and_protected_fixture_mutation_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="application-recheck-failures-") as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            shutil.copytree(evaluator.FIXTURE, baseline)
            baseline_report = evaluator.evaluate(baseline, root / "baseline-output")
            baseline_flags = dict(_flags(baseline_report))
            self.assertEqual(baseline_report["total"], 21)
            self.assertTrue(baseline_flags["rust_compiles"])
            self.assertFalse(baseline_report["all_pass"])

            invalid = root / "invalid-protected-fixture"
            _copy_reference(invalid)
            (invalid / "intention.json").write_text(
                '{"text":"changed","context":{}}\n', encoding="utf-8"
            )
            invalid_report = evaluator.evaluate(invalid, root / "invalid-output")
            invalid_flags = dict(_flags(invalid_report))
            self.assertFalse(invalid_flags["protected_files_intact"])
            self.assertFalse(invalid_flags["rust_compiles"])
            self.assertEqual(invalid_report["adapter_exits"], {})
            self.assertFalse(invalid_report["all_pass"])

    def test_missing_sandbox_fails_without_executing_candidate(self):
        with tempfile.TemporaryDirectory(prefix="application-recheck-sandbox-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            _copy_reference(workspace)
            with mock.patch.object(evaluator, "SANDBOX_EXEC", None):
                report = evaluator.evaluate(workspace, root / "output")
            flags = dict(_flags(report))
            self.assertFalse(flags["rust_compiles"])
            self.assertEqual(report["compile_exit"], 126)
            self.assertEqual(report["adapter_exits"], {})
            stderr = (root / "output" / "compile.stderr.txt").read_text()
            self.assertIn("sandbox-exec is unavailable", stderr)
            self.assertFalse((root / "output" / "candidate" / "observation-adapter").exists())

    def test_output_is_empty_or_create_only_and_outside_evidence(self):
        with tempfile.TemporaryDirectory(prefix="application-recheck-output-") as temporary:
            output = Path(temporary) / "occupied"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("existing\n", encoding="utf-8")
            with mock.patch.object(evaluator, "evaluate") as evaluate:
                with self.assertRaisesRegex(ValueError, "must be empty"):
                    recheck_application.run_recheck(EVIDENCE, output)
                evaluate.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing\n")

            with self.assertRaisesRegex(ValueError, "outside the recorded evidence"):
                recheck_application.run_recheck(EVIDENCE, EVIDENCE / "new-output")

    def test_nested_symlink_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory(
            prefix=".application-recheck-link-", dir=BENCHMARK
        ) as temporary:
            root = Path(temporary)
            evidence = root / "evidence"
            _copy_recorded_contract(evidence)
            external = root / "external.txt"
            external.write_text("unchanged\n", encoding="utf-8")
            (evidence / "control" / "project" / "src" / "ledger" / "link").symlink_to(
                external
            )
            with self.assertRaisesRegex(ValueError, "project tree contains a symlink"):
                recheck_application.run_recheck(evidence, root / "output")
            self.assertEqual(external.read_text(encoding="utf-8"), "unchanged\n")
            self.assertFalse((root / "output").exists())

    def test_reproduce_symlink_cannot_redirect_disposable_adaptation(self):
        with tempfile.TemporaryDirectory(
            prefix=".application-recheck-launcher-", dir=BENCHMARK
        ) as temporary:
            root = Path(temporary)
            evidence = root / "evidence"
            _copy_recorded_contract(evidence)
            external = root / "external-launcher.sh"
            external.write_text("do not overwrite\n", encoding="utf-8")
            launcher = evidence / "control" / "project" / "reproduce.sh"
            launcher.unlink()
            launcher.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "project tree contains a symlink"):
                recheck_application.run_recheck(evidence, root / "output")
            self.assertEqual(external.read_text(encoding="utf-8"), "do not overwrite\n")
            self.assertFalse((root / "output").exists())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO creation is unavailable")
    def test_special_file_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory(
            prefix=".application-recheck-special-", dir=BENCHMARK
        ) as temporary:
            root = Path(temporary)
            evidence = root / "evidence"
            _copy_recorded_contract(evidence)
            os.mkfifo(evidence / "control" / "project" / "unexpected-pipe")
            with self.assertRaisesRegex(ValueError, "project tree contains a special file"):
                recheck_application.run_recheck(evidence, root / "output")
            self.assertFalse((root / "output").exists())

    def test_symlinked_arm_ancestor_cannot_escape_evidence(self):
        with tempfile.TemporaryDirectory(
            prefix=".application-recheck-ancestor-", dir=BENCHMARK
        ) as temporary:
            root = Path(temporary)
            evidence = root / "evidence"
            _copy_recorded_contract(evidence)
            external = root / "outside-control"
            shutil.move(evidence / "control", external)
            (evidence / "control").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked component"):
                recheck_application.run_recheck(evidence, root / "output")
            self.assertTrue((external / "project" / "runtime.conf").is_file())
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
