from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import example
import ledger
import score


class ScoringTests(unittest.TestCase):
    def fixture(self, evaluator):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        bundle = example.create(root / 'bundle')
        checkout = root / 'checkout'
        checkout.mkdir()
        path = bundle / 'evaluator.py'
        path.write_text(evaluator)
        contract = ledger.read_json(bundle / 'contract.json')
        for case in contract['cases']:
            case['evaluator'] = {'path': path.name, 'sha256': ledger.digest(path)}
        (bundle / 'contract.json').write_text(json.dumps(contract))
        (bundle / 'contract.lock.json').unlink()
        ledger.freeze(bundle)
        events_path = bundle / 'events.jsonl'
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        events_path.write_text(''.join(json.dumps(e) + '\n' for e in events if e['kind'] != 'outcome'))
        return bundle, checkout

    def test_success_records_frozen_evaluator_and_refuses_duplicate(self):
        bundle, checkout = self.fixture('import os\nassert "TASK_SCORE_TEST_SECRET" not in os.environ\n')
        with patch.dict(os.environ, {'TASK_SCORE_TEST_SECRET': 'fixture-secret'}):
            receipt = score.score(bundle, 'native', 'cold-example', 0, checkout, trusted=True)
        self.assertEqual(receipt['status'], 'success')
        self.assertEqual(ledger.evaluate(bundle)['arms']['native']['successes'], 1)
        with self.assertRaises(ValueError):
            score.score(bundle, 'native', 'cold-example', 0, checkout, trusted=True)

    def test_failure_and_timeout_are_recorded(self):
        for program, status in [('raise SystemExit(2)', 'failure'), ('import time\ntime.sleep(10)', 'timeout')]:
            with self.subTest(status=status):
                bundle, checkout = self.fixture(program)
                receipt = score.score(bundle, 'native', 'cold-example', 0, checkout,
                                      trusted=True, timeout_seconds=1)
                self.assertEqual(receipt['status'], status)
                self.assertEqual(ledger.evaluate(bundle)['arms']['native']['outcome_counts'][status], 1)

    def test_untrusted_and_overlapping_paths_rejected_before_execution(self):
        bundle, checkout = self.fixture('raise RuntimeError("must not run")')
        with self.assertRaises(ValueError):
            score.score(bundle, 'native', 'cold-example', 0, checkout)
        with self.assertRaises(ValueError):
            score.score(bundle, 'native', 'cold-example', 0, bundle, trusted=True)
