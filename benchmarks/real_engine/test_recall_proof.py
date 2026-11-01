"""Audit-helper fixtures only; these are never benchmark evidence."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .recall_proof import (INITIAL_RECALL, RecallProofError, audit_relevance,
                          collect_recall_pages, fixed_ids, native_pairs, source_bindings, _run_item, _validate_item)
from .recall_proof import require_independent_stores


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
            result = _run_item(repo, repo / "runs", item, "unit-proof-only", Path(tmp) / "case")
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
                                     "unit-proof-only", result["engine"]["binary_sha256"])
            self.assertEqual(summary["returned_nodes"], 2)
            self.assertEqual(summary["relevant_memories"], 2)
            result["returned_nodes"] = 0
            with self.assertRaisesRegex(RecallProofError, "counts differ"):
                _validate_item(repo, repo / "runs", result, item, raw,
                               "unit-proof-only", result["engine"]["binary_sha256"])


if __name__ == "__main__":
    unittest.main()
