import unittest
from .runner import read_with_gate


class ReaderBoundaryTests(unittest.TestCase):
    def run_case(self, verdicts):
        calls, generations = [], []
        def tool(name, args):
            calls.append((name, args))
            if name == "session":
                return {"compile_receipt": {"offset": 9, "hash": "receipt"},
                        "compile_options": {"query": "Where is Mia now?"},
                        "reader_context": "[CONTROLLING] C: Mia lives in Denver"}
            return {"verdict": verdicts.pop(0)}
        def generate(context, question):
            generations.append(context)
            return "Mia lives in Denver"
        return tool, generate, calls, generations

    def test_normal_reader_generates_once(self):
        tool, generate, calls, generations = self.run_case(["Allow"])
        answer = read_with_gate(tool, generate, question="Where is Mia now?")
        self.assertIsInstance(answer, str)
        self.assertEqual(len(generations), 1)
        self.assertEqual(calls[1][1]["expected_receipt_hash"], "receipt")

    def test_stale_draft_retries_exactly_once(self):
        tool, generate, calls, generations = self.run_case([
            {"StaleLiteral": {"controlling": ["C"]}}, "Allow"])
        read_with_gate(tool, generate, question="Where is Mia now?")
        self.assertEqual(len(generations), 2)
        self.assertIn("C", generations[-1])

    def test_second_rejection_never_reaches_scorer(self):
        tool, generate, _, generations = self.run_case([
            {"FalseAbstention": {"controlling": ["C"]}},
            {"StaleLiteral": {"controlling": ["C"]}}])
        with self.assertRaises(RuntimeError):
            read_with_gate(tool, generate, question="Where is Mia now?")
        self.assertEqual(len(generations), 2)

    def test_abstention_comes_from_gate(self):
        tool, generate, _, _ = self.run_case([
            {"Abstain": {"answer": "not mentioned; no information available"}}])
        self.assertEqual(read_with_gate(tool, generate, question="Unknown?"),
                         "not mentioned; no information available")


if __name__ == "__main__":
    unittest.main()
