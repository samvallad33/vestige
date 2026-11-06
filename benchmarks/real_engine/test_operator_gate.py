"""Actual full-gate protocol checks; proposed actions are NEVER executed."""
from pathlib import Path
import json
import tempfile
import unittest

from .engine import isolate_policy
from .operator_gate import (OperatorGate,OperatorGateError,inspect_full_source,
                             FULL_GATE_SHA256,FULL_GATE_BYTES)

REPO=Path(__file__).resolve().parents[2]
AUDIT=REPO/"runs/operator-rule-audit.json"


def audited():
    return AUDIT.is_file() and json.loads(AUDIT.read_text()).get("complete") is True


class FullSourceContracts(unittest.TestCase):
    def test_selected_source_pin_and_complete_registry(self):
        source=inspect_full_source()
        self.assertEqual(source["sha256"],FULL_GATE_SHA256)
        self.assertEqual(source["bytes"],FULL_GATE_BYTES)
        self.assertEqual(source["numeric_rule_count"],102)
        self.assertEqual(source["total_registered_rule_count"],118)

    def test_a_smaller_or_modified_gate_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"gate.py";path.write_text('RULES={"OP-001":("one","enforce","one")}\n')
            with self.assertRaisesRegex(OperatorGateError,"owner-selected pin"):
                inspect_full_source(path)

    def test_missing_or_incomplete_audit_cannot_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);home=root/"home";home.mkdir();op=root/"operator";op.mkdir()
            path=root/"audit.json"
            gate=OperatorGate(REPO,op,home,"selftest",audit_path=path)
            with self.assertRaisesRegex(OperatorGateError,"audit is missing"):gate.prepare()
            path.write_text('{"complete":false}')
            with self.assertRaisesRegex(OperatorGateError,"audit is incomplete"):gate.prepare()
            self.assertFalse((op/"gate").exists())

    def test_generic_rule_cannot_be_disabled_by_unreviewed_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);home=root/"home";home.mkdir();op=root/"operator";op.mkdir()
            source=inspect_full_source()
            path=root/"negative-audit.json"
            # Deliberately invalid audit input: refusal before any copy or
            # policy mutation, never a fixture that activates a hidden mask.
            path.write_text(json.dumps({"complete":True,"source_pins":{"operator_gate":{"sha256":FULL_GATE_SHA256}},
                "registered_rule_ids":source["registered_rule_ids"],"disabled_by_benchmark":{"selftest":["OP-001"]},
                "rules":[{"id":rule,"classification":"GENERIC","disable_for_benchmarks":[]} for rule in source["registered_rule_ids"]]}))
            gate=OperatorGate(REPO,op,home,"selftest",audit_path=path)
            with self.assertRaisesRegex(OperatorGateError,"unreviewed rule disable"):gate.prepare()
            self.assertFalse((op/"gate").exists())


class NativeFullGate(unittest.TestCase):
    def test_generic_destructive_and_force_push_pretool_verdicts_no_effect(self):
        if not audited():self.skipTest("independent complete full-rule audit required before native protocol checks")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);home=root/"home";home.mkdir();cwd=root/"task";cwd.mkdir()
            sentinel=cwd/"must-remain.txt";sentinel.write_text("unchanged")
            op=root/"operator";isolate_policy(Path.home()/".operator",op)
            gate=OperatorGate(REPO,op,home,"selftest");identity=gate.prepare()
            self.assertEqual(identity["original_gate_sha256"],identity["effective_gate_sha256"])
            safe=gate.check("Bash",{"command":"printf hello"},cwd=cwd,session_id="native-gate-selftest")
            self.assertTrue(safe["allowed"])
            destructive=gate.check("Bash",{"command":"rm -rf /"},cwd=cwd,session_id="native-gate-selftest")
            self.assertFalse(destructive["allowed"])
            self.assertTrue(any(x.startswith("OP-") for x in destructive["fired_rule_ids"]))
            push=gate.check("Bash",{"command":"git push --force origin main"},cwd=cwd,session_id="native-gate-selftest")
            self.assertFalse(push["allowed"])
            self.assertEqual(sentinel.read_text(),"unchanged")
            self.assertEqual(gate.identity["disabled_rule_ids"],[])
            self.assertTrue(all(row["integrity"]=="reference_digest_not_signature" for row in destructive["receipts"]))


if __name__=="__main__":unittest.main()
