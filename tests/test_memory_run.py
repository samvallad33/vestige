"""Fake native/solver execution through real qualified guarded tool classes."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

import memory_run as runner
from campaigns.memoryagentbench.common import write_packet
from kernel_adapter import TOOL_NAMES, KernelBoundaryError


class FakeStack:
    made: ClassVar[list] = []
    fail_read = False
    override_read: ClassVar[dict | None] = None

    def __init__(self, root, *, profile):
        self.root, self.profile = Path(root), profile
        self.calls, self.nodes = [], []
        self.aborted = self.closed = False
        self.catalog = [
            {
                "name": name,
                "description": "fixture " + name,
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            }
            for name in sorted(TOOL_NAMES)
        ]
        self.made.append(self)

    def start(self, session):
        self.session = session

    def tool(self, session, name, args, call=None):
        self.calls.append((name, copy.deepcopy(args), call))
        if name == "smart_ingest":
            node = "mem-fixture-" + str(len(self.nodes) + 1)
            self.nodes.append(node)
            return {"structuredContent": {"nodeId": node}}
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
        if name == "recall":
            if self.override_read is not None:
                return copy.deepcopy(self.override_read)
            if self.fail_read:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": "fixture interdict"}],
                }
            return {
                "structuredContent": {
                    "retrieved": [{"content": "ONLY_DYNAMIC_SOURCE_BODY"}]
                }
            }
        return {"structuredContent": {"receipts": []}}

    def request(self, method, **kwargs):
        self.calls.append((method, copy.deepcopy(kwargs), None))
        if method == "ledger":
            return {"chain": {"status": "ok", "breaks": 0}}
        return {}

    def record(self, session, data):
        self.calls.append(("record", copy.deepcopy(data), None))
        return {}

    def stop(self, session):
        self.calls.append(("stop", session, None))

    def close(self):
        self.closed = True

    def abort(self):
        self.aborted = True


class FakeSolver:
    def __init__(self, *, never_finish=False):
        self.seen, self.never_finish = [], never_finish

    def query(self, query, runtime, env, messages, extra):
        self.seen.append(copy.deepcopy(messages))
        if messages[-1]["role"] == "tool" and not self.never_finish:
            message = {
                "role": "assistant",
                "content": [{"type": "text", "content": "fixture answer"}],
                "tool_calls": None,
            }
        else:
            function_call = __import__(
                "agentdojo.functions_runtime", fromlist=["FunctionCall"]
            ).FunctionCall
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    function_call(
                        function="vestige_recall",
                        args={"handle": "mem-fixture-1"},
                        id="fixture-call-" + str(len(self.seen)),
                    )
                ],
            }
        self.last_extra = extra
        return query, runtime, env, [*messages, message], extra


class MemoryRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Imports classes only; no solver/native factories or metadata calls.
        cls.qualified = runner.qualified_bindings()

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        FakeStack.made = []
        FakeStack.fail_read = False
        FakeStack.override_read = None
        self.solvers = []

    def prepared(self, *, gold_in_history=False):
        root = self.root / "prepared"
        history = [
            {
                "scope": "case0",
                "session_id": "s0",
                "ordinal": 0,
                "content": "SOURCE_BODY_NEVER_IN_HEADER",
            },
            {
                "scope": "case0",
                "session_id": "s1",
                "ordinal": 1,
                "content": "Later update.",
            },
        ]
        if gold_in_history:
            history[0]["answer"] = "ILLEGAL_GOLD"
        questions = [
            {"question_id": "q0", "scope": "case0", "question": "What changed?"},
            {"question_id": "q1", "scope": "case0", "question": "What changed later?"},
        ]
        write_packet(
            root,
            history,
            questions,
            [{"answer": "GOLD_SENTINEL_MUST_NOT_BE_OPENED"}],
            {"campaign": "memoryagentbench"},
        )
        return root

    def bindings(self, *, never_finish=False):
        def factory(logs_dir, model_name, **kwargs):
            self.assertEqual(model_name, runner.MODEL)
            self.assertLessEqual(kwargs["max_requests"], 40)
            solver = FakeSolver(never_finish=never_finish)
            self.solvers.append(solver)
            return solver

        return runner.Bindings(
            FakeStack,
            factory,
            lambda solver, model: {"provider_cost_usd": "0", "synthetic": True},
            self.qualified.runtime_factory,
            self.qualified.executor_factory,
            self.qualified.empty_env,
            self.qualified.text_block,
            synthetic=True,
        )

    def test_plan_never_constructs_native_or_solver_and_never_opens_gold(self):
        prepared = self.prepared()
        original = Path.open

        def guarded_open(path, *args, **kwargs):
            if path.name in {"oracle.jsonl", "answer-key.jsonl", "references.jsonl"}:
                raise AssertionError("gold file opened")
            return original(path, *args, **kwargs)

        with (
            mock.patch.object(Path, "open", guarded_open),
            mock.patch.object(
                runner,
                "qualified_bindings",
                side_effect=AssertionError("factory imported"),
            ),
        ):
            result = runner.run(prepared, "memoryagentbench", self.root / "plan")
        self.assertEqual(result["status"], "PLANNED_ZERO_MODEL_CALLS")
        self.assertEqual(len(result["selected_cases"]), 1)
        self.assertEqual(FakeStack.made, [])

    def test_full_dynamic_loop_uses_real_guarded_runtime_no_body_header_or_gold(self):
        prepared = self.prepared()
        original = Path.open

        def guarded_open(path, *args, **kwargs):
            if path.name == "oracle.jsonl":
                raise AssertionError("gold loaded into solver process")
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            result = runner.run(
                prepared,
                "memoryagentbench",
                self.root / "run",
                execute=True,
                bindings=self.bindings(),
            )
        self.assertEqual(result["status"], "SYNTHETIC_RUNNER_FIXTURES_COMPLETE")
        self.assertEqual(len(self.solvers[0].seen), 2)
        first = json.dumps(self.solvers[0].seen[0])
        self.assertNotIn("SOURCE_BODY_NEVER_IN_HEADER", first)
        self.assertNotIn("GOLD_SENTINEL", first)
        self.assertIn("mem-fixture-1", first)
        self.assertIn(
            "ONLY_DYNAMIC_SOURCE_BODY", runner.canonical(self.solvers[0].seen[1])
        )
        self.assertEqual(
            set(self.solvers[0].last_extra) >= {"_vestige_native_context"}, True
        )
        stack = FakeStack.made[0]
        self.assertEqual(stack.profile, "all_rules")
        self.assertTrue(stack.closed)
        self.assertIn("backfill", [call[0] for call in stack.calls])
        row = json.loads((self.root / "run/trial-00000/result.json").read_text())
        self.assertEqual(row["model_memory_reads"], 1)
        self.assertIsNone(row["official_score"])

    def test_every_question_uses_fresh_native_store_and_solver(self):
        runner.run(
            self.prepared(),
            "memoryagentbench",
            self.root / "two",
            execute=True,
            bounds=runner.RunBounds(limit=2),
            bindings=self.bindings(),
        )
        self.assertEqual(len(FakeStack.made), 2)
        self.assertNotEqual(FakeStack.made[0].session, FakeStack.made[1].session)
        self.assertEqual(len(self.solvers), 2)

    def test_write_bound_rejects_full_scope_before_any_effect(self):
        with self.assertRaisesRegex(ValueError, "no truncation"):
            runner.run(
                self.prepared(),
                "memoryagentbench",
                self.root / "bound",
                execute=True,
                bounds=runner.RunBounds(max_write_count=2),
                bindings=self.bindings(),
            )
        self.assertEqual(FakeStack.made, [])
        self.assertEqual(self.solvers, [])

    def test_model_iteration_limit_holds_and_preserves_no_retry(self):
        with self.assertRaisesRegex(runner.RunnerHeld, "iteration limit"):
            runner.run(
                self.prepared(),
                "memoryagentbench",
                self.root / "held",
                execute=True,
                bounds=runner.RunBounds(max_iters=1),
                bindings=self.bindings(never_finish=True),
            )
        self.assertEqual(len(self.solvers[0].seen), 1)
        self.assertTrue(FakeStack.made[0].aborted)
        self.assertTrue((self.root / "held/trial-00000/FAILED.json").exists())
        self.assertFalse((self.root / "held/predictions.jsonl").exists())

    def test_native_error_holds_before_second_solver_query(self):
        FakeStack.fail_read = True
        with self.assertRaisesRegex(runner.RunnerHeld, "invalid native tool response"):
            runner.run(
                self.prepared(),
                "memoryagentbench",
                self.root / "blocked",
                execute=True,
                bindings=self.bindings(),
            )
        self.assertEqual(len(self.solvers[0].seen), 1)
        self.assertTrue(FakeStack.made[0].aborted)

    def test_model_path_invalid_mcp_bodies_halt_before_second_query_or_prediction(self):
        values = [
            None,
            {},
            [],
            "scalar",
            {"error": "injected logical failure"},
            {"success": False},
            {"status": "FAIL_CLOSED_INTERDICTION"},
            {"status": "BLOCKED_BY_OPERATOR"},
        ]
        prepared = self.prepared()
        for index, body in enumerate(values):
            with self.subTest(body=body):
                FakeStack.override_read = {"structuredContent": body}
                out = self.root / ("invalid-mcp-" + str(index))
                before = len(self.solvers)
                with self.assertRaisesRegex(
                    runner.RunnerHeld, "invalid native tool response"
                ):
                    runner.run(
                        prepared,
                        "memoryagentbench",
                        out,
                        execute=True,
                        bindings=self.bindings(),
                    )
                self.assertEqual(len(self.solvers), before + 1)
                self.assertEqual(len(self.solvers[-1].seen), 1)
                self.assertTrue(FakeStack.made[-1].aborted)
                self.assertFalse((out / "predictions.jsonl").exists())
                self.assertTrue((out / "trial-00000/FAILED.json").exists())

    def test_nested_gold_history_rejected_before_write_and_query(self):
        with self.assertRaises(KernelBoundaryError):
            runner.run(
                self.prepared(gold_in_history=True),
                "memoryagentbench",
                self.root / "gold",
                execute=True,
                bindings=self.bindings(),
            )
        self.assertNotIn("smart_ingest", [row[0] for row in FakeStack.made[0].calls])
        self.assertEqual(self.solvers, [])

    def test_changed_runtime_bytes_rejected_before_factories(self):
        prepared = self.prepared()
        with (prepared / "history.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "runtime bytes changed"):
            runner.run(
                prepared,
                "memoryagentbench",
                self.root / "tampered",
                execute=True,
                bindings=self.bindings(),
            )
        self.assertEqual(FakeStack.made, [])

    def test_quotas_include_model_batches_and_controller_records_and_latch(self):
        stack = FakeStack(self.root, profile="all_rules")
        metered = runner.MeteredStack(stack, 2)
        metered.tool(
            "s", "smart_ingest", {"items": [{"content": "a"}, {"content": "b"}]}
        )
        with self.assertRaises(runner.RunnerHeld):
            metered.record("s", {"content": "c"})
        self.assertTrue(stack.aborted)
        with self.assertRaises(runner.RunnerHeld):
            metered.tool("s", "recall", {"handle": "mem"})

    def test_unknown_model_and_unbounded_iterations_rejected_without_metadata(self):
        for bounds in (
            runner.RunBounds(model="paid-model"),
            runner.RunBounds(max_iters=41),
            runner.RunBounds(max_write_count=10001),
        ):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                bounds.validate()

    def test_memform_public_packet_can_run_without_loading_answer_key(self):
        prepared = self.root / "memform"
        prepared.mkdir()
        packet = {
            "record_id": "mf-neutral",
            "context": {"domain": "fixture", "decision_variable": "choice"},
            "observations": [{"text": "SOURCE_BODY_NEVER_IN_HEADER"}],
            "direct_memory_text": None,
            "formation_instruction": "Learn observations",
            "tasks": [{"task": "Choose", "choices": ["A", "B"]}],
        }
        (prepared / "runtime-inputs.jsonl").write_text(runner.canonical(packet) + "\n")
        manifest = {"runtime_packets_sha256": runner.digest([packet])}
        manifest["sha256"] = runner.digest(manifest)
        (prepared / "manifest.json").write_text(runner.canonical(manifest))
        (prepared / "answer-key.jsonl").write_text("INVALID_GOLD_MUST_NEVER_BE_PARSED")
        result = runner.run(
            prepared,
            "memform",
            self.root / "mf-run",
            execute=True,
            bindings=self.bindings(),
        )
        self.assertEqual(result["completed"], 1)
        writes = [r for r in FakeStack.made[0].calls if r[0] == "smart_ingest"]
        self.assertEqual(writes[0][1]["content"], "SOURCE_BODY_NEVER_IN_HEADER")

    def test_longrca_full_public_trace_ingested_no_annotation_edges(self):
        prepared = self.root / "longrca"
        prepared.mkdir()
        public = {
            "question_ID": "terminal_bench_2__1",
            "history": [
                {
                    "step": 0,
                    "name": "worker",
                    "role": "assistant",
                    "content": "source event",
                }
            ],
        }
        path = prepared / "public.jsonl"
        path.write_text(runner.canonical(public) + "\n")
        (prepared / "manifest.json").write_text(
            runner.canonical(
                {"kind": "synthetic_fixture", "public_sha256": runner.file_sha(path)}
            )
        )
        result = runner.run(
            prepared,
            "longrca",
            self.root / "rca-run",
            execute=True,
            bindings=self.bindings(),
        )
        self.assertEqual(result["completed"], 1)
        write = next(row for row in FakeStack.made[0].calls if row[0] == "smart_ingest")
        self.assertEqual(write[1]["links"], [])
        self.assertNotIn("mistake_agent", write[1]["content"])


if __name__ == "__main__":
    unittest.main()
