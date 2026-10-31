"""Offline-only fake HTTP/native qualification. Never contacts Ollama."""

import base64
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

LIBRARY = Path(__file__).resolve().parents[1] / "upstream" / "agentdojo-library"
sys.path.insert(0, str(LIBRARY))
sys.path.insert(0, str(LIBRARY / "src"))

from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
from agentdojo.types import text_content_block_from_string
from vestige_eval.guarded_pipeline import AuditedSolver, native_function

from kernel_adapter import TOOL_NAMES
from local_solver import (
    ENDPOINT,
    PINNED_CONTEXT_MAX,
    PINNED_DIGEST,
    PINNED_MODEL,
    USAGE_KEY,
    HTTPResult,
    LocalAuditedSolver,
    LocalBounds,
    LocalBudget,
    LocalSolverError,
    LoopbackTransport,
    OllamaLLM,
    factory,
    inspect_model,
    validate_local_solver,
)


def metadata():
    return {
        "models": [
            {
                "name": PINNED_MODEL,
                "digest": PINNED_DIGEST,
                "size": 8988124298,
                "details": {
                    "format": "gguf",
                    "family": "qwen2",
                    "parameter_size": "14.8B",
                    "quantization_level": "Q4_K_M",
                    "context_length": PINNED_CONTEXT_MAX,
                },
                "capabilities": ["completion", "tools", "insert"],
            }
        ]
    }


def response(content="complete", calls=None, thinking="retained reasoning"):
    message = {"role": "assistant", "content": content, "thinking": thinking}
    if calls is not None:
        message["tool_calls"] = calls
    return {
        "model": PINNED_MODEL,
        "message": message,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 20,
    }


def tool_call(name="vestige_recall", *, call_id=None):
    call = {"function": {"index": 0, "name": name, "arguments": {"tag": "history"}}}
    if call_id is not None:
        call["id"] = call_id
    return call


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def exchange(self, method, path, body=None, **kwargs):
        self.calls.append((method, path, body, kwargs))
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return (
            result
            if isinstance(result, HTTPResult)
            else HTTPResult(200, json.dumps(result).encode())
        )


class FakeStack:
    def __init__(self, *, allowed=True, drop_after=False):
        self.calls = []
        self.allowed, self.drop_after = allowed, drop_after

    def request(self, method, **kwargs):
        self.calls.append((method, copy.deepcopy(kwargs)))
        if method == "before":
            return {"allowed": self.allowed}
        if self.drop_after:
            raise OSError("native after RPC dropped")
        return {}  # Qualified bridge's successful after() has no payload.

    def record(self, session, data):
        self.calls.append(("record", {"session": session, "data": copy.deepcopy(data)}))


class LocalSolverTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.runtime = FunctionsRuntime(
            [
                native_function(
                    {
                        "name": name,
                        "description": "original native schema " + name,
                        "inputSchema": {
                            "type": "object",
                            "properties": {"tag": {"type": "string"}},
                            "additionalProperties": False,
                        },
                    }
                )
                for name in sorted(TOOL_NAMES)
            ]
        )
        self.messages = [
            {
                "role": "user",
                "content": [text_content_block_from_string("Check original trace.")],
            }
        ]

    def solver(self, *results, bounds=None):
        transport = FakeTransport(*results)
        llm = OllamaLLM(transport, LocalBudget(bounds or LocalBounds()))
        return LocalAuditedSolver(llm, Path(self.folder.name)), transport

    def query(self, solver, messages=None, extra=None):
        return solver.query(
            "Check original trace.",
            self.runtime,
            EmptyEnv(),
            self.messages if messages is None else messages,
            extra,
        )

    def tool_result(self, result, *, identifier=None):
        call = result[-2][-1]["tool_calls"][0]
        return {
            "role": "tool",
            "tool_call": call,
            "tool_call_id": identifier or call.id,
            "content": [
                text_content_block_from_string("original dynamic memory output")
            ],
            "error": None,
        }

    def test_exact_nineteen_tool_schemas_forwarded_and_local_usage_recorded(self):
        solver, transport = self.solver(response())
        result = self.query(solver)
        request = json.loads(transport.calls[0][2])
        self.assertEqual(len(request["tools"]), 19)
        self.assertEqual(
            {x["function"]["name"] for x in request["tools"]},
            set(self.runtime.functions),
        )
        for tool in request["tools"]:
            function = self.runtime.functions[tool["function"]["name"]]
            self.assertEqual(
                tool["function"]["parameters"], function.parameters.model_json_schema()
            )
        self.assertNotIn("think", request)
        self.assertFalse(request["stream"])
        self.assertEqual(request["options"]["num_ctx"], 32768)
        usage = result[-1][USAGE_KEY][0]["usage"]
        self.assertEqual(
            usage,
            {"prompt_eval_count": 100, "eval_count": 20, "provider_cost_usd": "0"},
        )
        self.assertNotIn("openai", json.dumps(request).lower())
        self.assertIsInstance(solver, AuditedSolver)

    def test_serial_dynamic_tool_loop_preserves_thinking_provider_and_result_ids(self):
        first = response("", [tool_call(call_id="native-provider-id")])
        solver, transport = self.solver(first, response())
        result = self.query(solver, extra={"session_id": "task-1", "run_id": "run-1"})
        current = [*result[-2], self.tool_result(result)]
        self.query(solver, current, result[-1])
        request = json.loads(transport.calls[1][2])
        self.assertEqual(request["messages"][-2], first["message"])
        self.assertEqual(request["messages"][-1]["tool_call_id"], "native-provider-id")
        self.assertEqual(request["messages"][-1]["tool_name"], "vestige_recall")
        self.assertEqual(
            request["messages"][-1]["content"], "original dynamic memory output"
        )
        self.assertEqual(
            result[-1][USAGE_KEY][0]["tool_identity"][0]["identity_origin"], "provider"
        )

    def test_missing_provider_ids_are_explicit_generated_not_falsely_attributed(self):
        solver, transport = self.solver(response("", [tool_call()]), response())
        result = self.query(solver)
        identity = result[-1][USAGE_KEY][0]["tool_identity"][0]
        self.assertEqual(identity["identity_origin"], "adapter_generated")
        self.assertIsNone(identity["provider_id"])
        self.assertTrue(identity["call_id"].startswith("ollama-"))
        self.query(solver, [*result[-2], self.tool_result(result)], result[-1])
        self.assertNotIn(
            "tool_call_id", json.loads(transport.calls[1][2])["messages"][-1]
        )

    def test_wrong_result_id_halts_before_second_transport(self):
        solver, transport = self.solver(response("", [tool_call()]))
        result = self.query(solver)
        with self.assertRaises(LocalSolverError):
            self.query(
                solver,
                [*result[-2], self.tool_result(result, identifier="wrong")],
                result[-1],
            )
        self.assertEqual(len(transport.calls), 1)
        self.assertTrue(solver.solver.failed)

    def test_changed_history_session_and_cross_instance_state_are_rejected(self):
        for change in ("history", "session", "instance"):
            with self.subTest(change=change):
                solver, transport = self.solver(response("", [tool_call()]))
                result = self.query(solver, extra={"session_id": "a"})
                current = [*result[-2], self.tool_result(result)]
                extra = result[-1]
                if change == "history":
                    current[0] = dict(
                        current[0], content=[text_content_block_from_string("changed")]
                    )
                elif change == "session":
                    extra = {**extra, "session_id": "b"}
                else:
                    solver, transport = self.solver(response())
                before = len(transport.calls)
                with self.assertRaises(LocalSolverError):
                    self.query(solver, current, extra)
                self.assertEqual(len(transport.calls), before)

    def test_unknown_parallel_and_replayed_calls_fail_closed(self):
        for calls in ([tool_call("unavailable")], [tool_call(), tool_call()]):
            solver, transport = self.solver(response("", calls))
            with self.assertRaises(LocalSolverError):
                self.query(solver)
            self.assertEqual(len(transport.calls), 1)
        solver, transport = self.solver(
            response("", [tool_call(call_id="same")]),
            response("", [tool_call(call_id="same")]),
        )
        result = self.query(solver)
        with self.assertRaises(LocalSolverError):
            self.query(solver, [*result[-2], self.tool_result(result)], result[-1])

    def test_empty_malformed_nonfinite_duplicate_and_truncated_responses_fail_closed(
        self,
    ):
        bad = []
        bad.append(HTTPResult(200, b'{"a":1,"a":2}'))
        bad.append(HTTPResult(200, b'{"bad":NaN}'))
        bad.append(HTTPResult(200, b""))
        bad.append(response("", [], "thinking alone is not a completed answer"))
        bad.append({**response(), "done_reason": "length"})
        bad.append({**response(), "done": False})
        bad.append({**response(), "prompt_eval_count": None})
        bad.append(response("nonempty but malformed tool_calls", {}))
        for payload in bad:
            with self.subTest(payload=payload):
                solver, transport = self.solver(payload)
                with self.assertRaises(LocalSolverError):
                    self.query(solver)
                with self.assertRaises(LocalSolverError):
                    self.query(solver)
                self.assertEqual(len(transport.calls), 1)
                self.assertTrue(solver.solver.budget.failed)

    def test_transport_unknown_is_not_retried_and_accounting_stays_uncertain(self):
        solver, transport = self.solver(TimeoutError("unknown local generation state"))
        with self.assertRaises(LocalSolverError):
            self.query(solver)
        self.assertEqual(
            solver.solver.budget.records[0]["status"], "transport_uncertain"
        )
        with self.assertRaises(LocalSolverError):
            self.query(solver)
        self.assertEqual(len(transport.calls), 1)

    def test_request_limit_and_oversized_history_refuse_before_transport(self):
        solver, transport = self.solver(response(), bounds=LocalBounds(max_requests=1))
        self.query(solver)
        with self.assertRaises(LocalSolverError):
            self.query(solver)
        self.assertEqual(len(transport.calls), 1)
        solver, transport = self.solver(response())
        large = [
            {"role": "user", "content": [text_content_block_from_string("雪" * 20000)]}
        ]
        with self.assertRaises(LocalSolverError):
            self.query(solver, large)
        self.assertEqual(transport.calls, [])

    def test_usage_breach_records_measured_counts_then_halts(self):
        solver, _transport = self.solver({**response(), "eval_count": 8193})
        with self.assertRaises(LocalSolverError):
            self.query(solver)
        self.assertEqual(solver.solver.budget.records[0]["status"], "bound_breached")
        self.assertEqual(solver.solver.budget.records[0]["usage"]["eval_count"], 8193)

    def test_native_audit_has_truthful_provider_exact_request_and_raw_response(self):
        payload = response()
        solver, transport = self.solver(payload)
        stack = FakeStack()
        self.query(
            solver,
            extra={"session_id": "native-a", "_vestige_native_context": (stack, "sid")},
        )
        before, after, record = stack.calls
        self.assertEqual(record[1]["data"]["provider"], "ollama_chat")
        self.assertEqual(before[0], "before")
        self.assertEqual(after[0], "after")
        self.assertEqual(before[1]["tool"], "ollama_chat")
        self.assertEqual(before[1]["call"], after[1]["call"])
        self.assertEqual(before[1]["args"], after[1]["args"])
        self.assertEqual(
            before[1]["args"]["request"], json.loads(transport.calls[0][2])
        )
        actual = base64.b64decode(after[1]["result"]["response_body_b64"])
        self.assertEqual(json.loads(actual), payload)
        self.assertEqual(
            hashlib.sha256(actual).hexdigest(), after[1]["result"]["response_sha256"]
        )
        events = [
            json.loads(x)
            for x in (Path(self.folder.name) / "native-a/provider.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(
            [x["event"] for x in events],
            ["ollama_request", "ollama_response", "ollama_usage"],
        )
        self.assertTrue(all(x["provider"] == "ollama_chat" for x in events))
        self.assertEqual(
            base64.b64decode(events[0]["request_body_b64"]), transport.calls[0][2]
        )

    def test_operator_denial_and_dropped_after_stop_with_no_retry(self):
        for allowed, drop in ((False, False), (True, True)):
            solver, transport = self.solver(response())
            stack = FakeStack(allowed=allowed, drop_after=drop)
            with self.assertRaises((LocalSolverError, OSError)):
                self.query(
                    solver,
                    extra={
                        "session_id": "deny-test",
                        "_vestige_native_context": (stack, "sid"),
                    },
                )
            self.assertEqual(len(transport.calls), 1 if allowed else 0)
            self.assertEqual(
                sum(x[0] == "after" for x in stack.calls), 1 if allowed else 0
            )

    def test_loopback_http_ignores_proxy_env_and_never_follows_redirects(self):
        connection = mock.Mock()
        connection.getresponse.return_value.status = 302
        connection.getresponse.return_value.read.return_value = b"redirect"
        with (
            mock.patch.dict(
                "os.environ",
                {"HTTP_PROXY": "https://evil.invalid", "OPENAI_API_KEY": "DO_NOT_SEND"},
            ),
            mock.patch(
                "local_solver.http.client.HTTPConnection", return_value=connection
            ) as make,
        ):
            result = LoopbackTransport().exchange(
                "POST", "/api/chat", b"{}", timeout=12
            )
        make.assert_called_once_with("127.0.0.1", 11434, timeout=12)
        connection.request.assert_called_once_with(
            "POST",
            "/api/chat",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(result.status, 302)
        connection.close.assert_called_once()
        with self.assertRaises(LocalSolverError):
            LoopbackTransport().exchange("POST", "https://evil.invalid")

    def test_factory_is_metadata_only_explicit_model_and_validator_blocks_swaps(self):
        tags = HTTPResult(200, json.dumps(metadata()).encode())
        with mock.patch.object(
            LoopbackTransport, "exchange", return_value=tags
        ) as exchange:
            solver = factory(self.folder.name, PINNED_MODEL)
        self.assertEqual(exchange.call_count, 1)
        self.assertEqual(exchange.call_args.args, ("GET", "/api/tags"))
        contract = validate_local_solver(solver, model_name=PINNED_MODEL)
        self.assertEqual(contract["endpoint"], ENDPOINT)
        self.assertEqual(contract["provider_cost_usd"], "0")
        solver.solver.model = "other"
        with self.assertRaises(LocalSolverError):
            validate_local_solver(solver, PINNED_MODEL)
        with self.assertRaises(ValueError):
            factory(self.folder.name, None)
        with self.assertRaises(ValueError):
            factory(self.folder.name, PINNED_MODEL, endpoint="https://external")

    def test_metadata_mismatch_absence_and_bounds_cannot_silently_fallback(self):
        for change in ("missing", "digest", "quant", "context", "capabilities"):
            data = metadata()
            if change == "missing":
                data["models"] = []
            elif change == "digest":
                data["models"][0]["digest"] = "different"
            elif change == "quant":
                data["models"][0]["details"]["quantization_level"] = "Q8_0"
            elif change == "context":
                del data["models"][0]["details"]["context_length"]
            else:
                del data["models"][0]["capabilities"]
            with self.assertRaises(LocalSolverError):
                inspect_model(FakeTransport(data))
        for kwargs in (
            {"max_requests": 41},
            {"max_requests": True},
            {"context_tokens": 32769},
            {"max_output_tokens": 60000},
            {"max_input_tokens": 32769},
            {"input_overhead_tokens": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LocalBudget(LocalBounds(**kwargs))


if __name__ == "__main__":
    unittest.main()
