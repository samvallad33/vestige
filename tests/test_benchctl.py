"""Control-plane checks; no benchmark, model or native process is launched."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchctl


class OrchestratorTests(unittest.TestCase):
    def test_output_cannot_escape_evidence_or_replace_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(benchctl, "ROOT", root):
                with self.assertRaises(ValueError):
                    benchctl.create_output(root / "outside")
                path = benchctl.create_output(root / "evidence/run")
                with self.assertRaises(FileExistsError):
                    benchctl.create_output(path)

    def run_fixture(self, process, *, identity=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "campaigns/longrca/prepare.py"
            source.parent.mkdir(parents=True)
            source.write_text("# Fixture adapter; never imported\n")
            with (
                patch.object(benchctl, "ROOT", root),
                patch.object(
                    benchctl,
                    "source_identity",
                    side_effect=identity or [{"source": "a"}, {"source": "a"}],
                ),
                patch.object(
                    benchctl.subprocess, "run", side_effect=process
                ) as execute,
            ):
                report = benchctl.prepare_all(
                    root / "evidence/run", selected=["longrca"], limit=2
                )
                persisted = json.loads((root / "evidence/run/suite.json").read_text())
                self.assertEqual(persisted, report)
                self.assertEqual(execute.call_count, 1)
                return report, execute.call_args

    def test_preparation_failure_is_retained_without_retry(self):
        report, call = self.run_fixture(
            [subprocess.CompletedProcess([], 9, "partial", "fixture error")]
        )
        self.assertEqual(report["campaigns"][0]["status"], "PREPARATION_FAILED")
        self.assertEqual(report["campaigns"][0]["exit_code"], 9)
        self.assertIsNone(call.kwargs.get("shell"))
        self.assertEqual(report["provider_calls"], 0)

    def test_timeout_is_preserved_without_automatic_retry(self):
        report, _ = self.run_fixture(
            [subprocess.TimeoutExpired(["fixture"], 1, output=b"partial")]
        )
        self.assertEqual(report["campaigns"][0]["status"], "PREPARATION_TIMEOUT")

    def test_source_drift_is_not_prepared_identity(self):
        report, _ = self.run_fixture(
            [subprocess.CompletedProcess([], 0, "", "")],
            identity=[{"source": "a"}, {"source": "b"}],
        )
        self.assertEqual(report["state"], "SOURCE_DRIFT")

    def test_invalid_selection_fails_before_output_or_process(self):
        with patch.object(benchctl.subprocess, "run") as execute:
            with self.assertRaises(ValueError):
                benchctl.prepare_all("unused", selected=["not-a-campaign"])
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
