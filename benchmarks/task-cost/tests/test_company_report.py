from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import company_report
import example
import ledger


class CompanyReportTests(unittest.TestCase):
    def test_manifest_summary_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = example.create(root / "bundle")
            output = company_report.create(
                bundle, root / "report", "native", "candidate"
            )
            self.assertEqual(
                {p.name for p in output.iterdir()},
                {"manifest.json", "REPORT.md", "comparison.json", "qualification.json"},
            )
            manifest = ledger.read_json(output / "manifest.json")
            for name, digest in manifest["files"].items():
                self.assertEqual(ledger.digest(output / name), digest)
            self.assertIn(
                "Synthetic evidence validates accounting only",
                (output / "REPORT.md").read_text(),
            )
            with self.assertRaises(ValueError):
                company_report.create(bundle, output, "native", "candidate")
