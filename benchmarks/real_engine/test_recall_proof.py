"""Audit-helper fixtures only; these are never benchmark evidence."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

from .recall_proof import (INITIAL_RECALL, RecallProofError, audit_relevance,
                          collect_recall_pages, fixed_ids, native_pairs, source_bindings, _run_item, _validate_item)
from .recall_proof import require_independent_stores
from .recall_proof import audit_hook_models


def node(identity="mem-test-one", text="[date] user: original raw source"):
    return {"id": identity, "content": text}


def page(nodes, following=None):
    out = {"handle": "history", "kind": "tag", "nodes": nodes}
    if following is not None:
        out["next"] = following
    return out


class RecallPaginationTests(unittest.TestCase):
    def test_only_generic_handle_and_native_cursor_are_used(self):
        following = {"handle": "history", "offset": 1, "limit": 100, "detail_level": "full"}
        outputs = iter([page([node()], following), page([node("mem-test-two", "other")])])
        requests = []
        def call(name, arguments):
            requests.append((name, deepcopy(arguments)))
            return next(outputs)
        pages, nodes = collect_recall_pages(call, max_pages=3)
        self.assertEqual(requests, [("recall", INITIAL_RECALL), ("recall", following)])
        self.assertEqual(len(pages), 2)
        self.assertEqual(len(nodes), 2)

    def test_empty_error_repeated_ids_and_skipped_cursor_fail(self):
        cases = [page([]), {"error": "handle_required"},
                 page([node(), node()]), page([node()], {"handle": "history", "offset": 2, "detail_level": "full"})]
        for result in cases:
            with self.subTest(result=result), self.assertRaises(RecallProofError):
                collect_recall_pages(lambda *_: result, max_pages=2)

    def test_schema_unrelated_handle_and_nontermination_fail(self):
        for result in [{"handle": "oracle", "kind": "tag", "nodes": [node()]},
                       page([{"id": "id-without-content"}])]:
            with self.assertRaises(RecallProofError):
                collect_recall_pages(lambda *_: result, max_pages=2)
        following = {"handle": "history", "offset": 1, "limit": 100, "detail_level": "full"}
        with self.assertRaisesRegex(RecallProofError, "did not terminate"):
            collect_recall_pages(lambda *_: page([node()], following), max_pages=1)


class SourceBindingTests(unittest.TestCase):
    def test_store_directories_must_be_independent_but_os_pid_reuse_is_valid(self):
        entries = [{"root": "/fixture/one", "engine": {"data_dir": "/fixture/one/native/strata", "pid": 10}},
                   {"root": "/fixture/two", "engine": {"data_dir": "/fixture/two/native/strata", "pid": 10}}]
        require_independent_stores(entries)
        for field in ("root", "data_dir"):
            duplicate = deepcopy(entries)
            if field == "root":
                duplicate[1]["root"] = duplicate[0]["root"]
            else:
                duplicate[1]["engine"]["data_dir"] = duplicate[0]["engine"]["data_dir"]
            with self.assertRaisesRegex(RecallProofError, "reused"):
                require_independent_stores(duplicate)

    def pair(self, content="raw", identity="mem-test-one"):
        request = {"kind": "tool_request", "name": "smart_ingest", "arguments": {
            "content": content, "node_type": "note", "tags": ["history", "session-0"], "source": "history:0"}}
        reply = {"kind": "tool_response", "name": "smart_ingest", "result": {"nodeId": identity}}
        return (1, request, 2, reply)

    def test_actual_returned_id_and_original_bytes_are_required(self):
        bound = source_bindings(["raw"], [self.pair()], [node(text="raw")])
        self.assertEqual(bound["mem-test-one"]["content"], "raw")
        for returned in [[node(text="substituted")], [node("another-store-id", "raw")], []]:
            with self.assertRaises(RecallProofError):
                source_bindings(["raw"], [self.pair()], returned)

    def test_native_reinforcement_links_same_item_target_and_echo(self):
        first = self.pair()
        first[3]["result"]["canonicalHash"] = "a" * 64
        second = deepcopy(first)
        second[1]["arguments"].update(tags=["history", "session-1"], source="history:1")
        second[3]["result"] = {"success": True, "decision": "reinforce",
            "duplicateOf": "mem-test-one", "echoNodeId": "mem-test-echo",
            "canonicalHash": "a" * 64, "receiptId": "eff-test"}
        echo = "duplicate of mem-test-one\ncanonical_hash: " + "a" * 64 + "\nsubmitted_line_count: 1"
        nodes = [node(text="raw"), node("mem-test-echo", echo)]
        bound = source_bindings(["raw", "raw"], [first, second], nodes)
        self.assertEqual(bound["mem-test-echo"]["duplicate_of"], "mem-test-one")
        for field, value in [("success", False), ("duplicateOf", "foreign"),
                             ("echoNodeId", None), ("canonicalHash", "b" * 64)]:
            bad = deepcopy(second); bad[3]["result"][field] = value
            with self.assertRaises(RecallProofError):
                source_bindings(["raw", "raw"], [first, bad], nodes)
        with self.assertRaises(RecallProofError):
            source_bindings(["raw", "raw"], [first, second], nodes[:1])

    def test_changed_admission_or_missing_node_id_fails(self):
        for mutation in [lambda pair: pair[1]["arguments"].update(forceCreate=True),
                         lambda pair: pair[3]["result"].clear()]:
            pair = self.pair(); mutation(pair)
            with self.assertRaises(RecallProofError):
                source_bindings(["raw"], [pair], [node(text="raw")])
        pair = self.pair()
        pair[1]["arguments"] = {"items": [pair[1]["arguments"]], "role": "system"}
        pair[3]["result"] = {"results": [{"nodeId": "mem-test-one"}]}
        with self.assertRaises(RecallProofError):
            source_bindings(["raw"], [pair], [node(text="raw")])

    def test_native_request_reply_errors_and_unmatched_requests_fail(self):
        for events in [[{"kind": "tool_request", "name": "recall"}],
                       [{"kind": "tool_response", "name": "recall", "result": {}}],
                       [{"kind": "tool_request", "name": "recall"}, {"kind": "tool_error", "name": "recall"}]]:
            with self.assertRaises(RecallProofError):
                native_pairs(events)

    def test_only_exact_later_forwarded_hook_error_is_accepted(self):
        raw = {"isError": True, "content": [{"type": "text", "text": "unsupported native lookup"}]}
        request = {"kind": "tool_request", "name": "recall", "arguments": {"mode": "reason"}}
        error = {"kind": "tool_error", "name": "recall", "timestamp": 10, "raw_mcp_result": raw}
        visible = {"name": "recall", "arguments": {"mode": "reason"}, "result": raw, "seen_at": 11}
        self.assertEqual(native_pairs([request, error], [visible])[0][3]["result"], raw)
        for mutation in [dict(visible, seen_at=9), dict(visible, arguments={"mode": "lookup"}),
                         dict(visible, result={"isError": True, "content": []})]:
            with self.assertRaises(RecallProofError):
                native_pairs([request, error], [mutation])


class MeteredHookProofTests(unittest.TestCase):
    def test_actual_response_usage_caps_and_shared_ledger_are_required(self):
        from .provider import MODEL, account
        from .runtime import BudgetLedger, append_jsonl
        from .recall_proof import ITEM_LIMITS
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp); root = runs / "proof"; root.mkdir()
            identity = {"benchmark": "longmemeval", "arm": "vestige", "item_id": "unit",
                        "config_hash": "config", "phase": "hook"}
            result = {"usage": {"prompt_tokens": 5, "completion_tokens": 2},
                      "choices": [{"message": {"role": "assistant", "content": "fixture"}}]}
            cost, usage = account(result)
            request = {"event": "request", "call_id": "unit-hook-call", "identity": identity,
                "payload": {"model": MODEL, "temperature": 0, "max_completion_tokens": 512,
                            "messages": [{"role": "user", "content": "neutral fixture"}]}}
            response = {"event": "response", "call_id": "unit-hook-call", "identity": identity,
                        "response": result, "usage": usage, "cost": cost}
            append_jsonl(root / "hook-api.jsonl", request)
            append_jsonl(root / "hook-api.jsonl", response)
            append_jsonl(root / "budget.jsonl", {"event": "budget_after", "limits": ITEM_LIMITS, "usage": usage})
            ledger = BudgetLedger(runs / "budget.db")
            ledger.reserve("unit-hook-call/attempt-0", 0.01)
            with self.assertRaisesRegex(RecallProofError, "settled"):
                audit_hook_models(root, runs, "unit", "config")
            ledger.settle("unit-hook-call/attempt-0", cost, usage)
            checked = audit_hook_models(root, runs, "unit", "config")
            self.assertEqual(checked["successful_calls"], 1)
            self.assertEqual(checked["cost"], cost)
            self.assertEqual(checked["usage"], [usage])
            request["payload"]["model"] = "wrong-model"
            (root / "hook-api.jsonl").write_text(json.dumps(request) + "\n" + json.dumps(response) + "\n")
            with self.assertRaisesRegex(RecallProofError, "different model"):
                audit_hook_models(root, runs, "unit", "config")


class OfflineRelevanceTests(unittest.TestCase):
    def setUp(self):
        self.raw = {"haystack_session_ids": ["distractor", "official-target"],
            "haystack_dates": ["first-date", "second-date"],
            "haystack_sessions": [[{"role": "user", "content": "other"}],
                [{"role": "user", "content": "the actual evidence", "has_answer": True},
                 {"role": "assistant", "content": "acknowledged"}]],
            "answer_session_ids": ["official-target"]}

    def test_relevance_uses_exact_returned_source_bytes_only_after_retrieval(self):
        records = [node(text="[second-date] user: the actual evidence")]
        result = audit_relevance(self.raw, records)
        self.assertEqual(result[0]["node_id"], "mem-test-one")
        self.assertEqual(result[0]["basis"], "official has_answer turn")
        with self.assertRaises(RecallProofError):
            audit_relevance(self.raw, [node(text="the actual evidence")])

    def test_missing_relevant_session_or_label_fails(self):
        for target in ["missing-session", "official-target"]:
            raw = deepcopy(self.raw); raw["answer_session_ids"] = [target]
            with self.assertRaises(RecallProofError):
                audit_relevance(raw, [node(text="[first-date] user: other")])
        raw = deepcopy(self.raw); raw["answer_session_ids"] = []
        with self.assertRaises(RecallProofError):
            audit_relevance(raw, [])

    def test_fixed_official_sample_is_exactly_ten_and_no_shrunk_manifest(self):
        items = [SimpleNamespace(id=str(index)) for index in range(500)]
        self.assertEqual(len(fixed_ids(items)), 10)
        self.assertEqual(fixed_ids(items), fixed_ids(items))
        with self.assertRaises(RecallProofError):
            fixed_ids(items[:10])


@unittest.skipUnless((Path(__file__).resolve().parents[2] / "runs/build-manifest.json").is_file(),
                     "actual release build required")
class ActualNativeProofHelperTests(unittest.TestCase):
    def test_real_child_pages_source_bindings_negative_and_kill(self):
        from .runtime import read_jsonl
        repo = Path(__file__).resolve().parents[2]
        # A neutral local protocol fixture, never an official/scored item.
        item = SimpleNamespace(id="unit-proof-only", question="What is in the fixture?", history=[
            "[fixture-date] user: Native proof fixture: the calendar says Thursday.",
            "[fixture-date] user: Native proof fixture: the meeting room is Cedar."])
        with tempfile.TemporaryDirectory(dir=repo / "runs") as tmp:
            result = _run_item(repo, repo / "runs", item, "unit-proof-only", Path(tmp) / "case",
                               hook_profile="memory_qa")
            self.assertEqual(result["status"], "PASS", result.get("error"))
            events = read_jsonl(Path(result["native_log"]))
            pairs = native_pairs(events)
            replies = [reply["result"] for _, request, _, reply in pairs
                       if request["name"] == "recall" and request["arguments"].get("handle") == "history"]
            nodes = [node for reply in replies for node in reply["nodes"]]
            self.assertEqual(len(source_bindings(item.history, pairs, nodes)), 2)
            self.assertEqual(len(nodes), 2)
            self.assertEqual(sum(e.get("kind") == "proof_runner_aborted" for e in events), 1)
            raw = {"haystack_session_ids": ["fixture-session"], "haystack_dates": ["fixture-date"],
                "haystack_sessions": [[{"role": "user", "content": text.split("user: ", 1)[1],
                                        "has_answer": True} for text in item.history]],
                "answer_session_ids": ["fixture-session"]}
            result["relevant_memories"] = audit_relevance(raw, nodes)
            summary = _validate_item(repo, repo / "runs", result, item, raw,
                                     "unit-proof-only", result["engine"]["binary_sha256"],
                                     required_hook_profile="memory_qa")
            self.assertEqual(summary["returned_nodes"], 2)
            self.assertEqual(summary["relevant_memories"], 2)
            result["returned_nodes"] = 0
            with self.assertRaisesRegex(RecallProofError, "counts differ"):
                _validate_item(repo, repo / "runs", result, item, raw,
                               "unit-proof-only", result["engine"]["binary_sha256"],
                               required_hook_profile="memory_qa")


if __name__ == "__main__":
    unittest.main()
