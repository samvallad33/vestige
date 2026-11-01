"""Zero-inference native qualification. Fixtures are not scored AgentDojo tasks."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from campaigns.terminal_bench.native_vendor.stack import NativeStack, body

EVENTS = {
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PreCompact",
    "PostCompact",
    "SubagentStart",
    "SubagentStop",
    "Stop",
    "SessionEnd",
}


def verify_receipt(receipt):
    verification = receipt["attestation"]["verification"]
    assert (
        verification["chainValid"]
        and verification["gateAllowed"]
        and verification["locallyVerified"]
    )


def verify_review(before, review, after, action):
    first, last = before["node"], after["node"]
    assert review["nodeId"] == first["id"] == last["id"]
    assert last["reps"] == first["reps"] + 1
    assert last["lapses"] == first["lapses"] + (action == "demote")
    for key in ["reps", "lapses", "difficulty", "stability"]:
        assert review["changes"][key]["before"] == first[key]
        assert review["changes"][key]["after"] == last[key]


def verify_hooks(acks, sid, child):
    observed = {r["event_name"] for r in acks}
    assert EVENTS <= observed, sorted(EVENTS - observed)
    assert all(
        r["current_window_verified"] and r["historical_breaks"] == 0 for r in acks
    )
    assert all(r["session"] in {sid, child} for r in acks)
    for event in ["SubagentStart", "SubagentStop"]:
        found = [r for r in acks if r["event_name"] == event]
        assert (
            len(found) == 1
            and found[0]["session"] == sid
            and found[0]["agent"] == child
        )
    for session in [sid, child]:
        for event in ["SessionStart", "Stop", "SessionEnd"]:
            assert any(
                r["session"] == session and r["event_name"] == event for r in acks
            )
    for event in ["PreCompact", "PostCompact", "UserPromptSubmit"]:
        assert any(r["session"] == sid and r["event_name"] == event for r in acks)
    pres = {(r["session"], r["call"]) for r in acks if r["event_name"] == "PreToolUse"}
    posts = {
        (r["session"], r["call"])
        for r in acks
        if r["event_name"] in {"PostToolUse", "PostToolUseFailure"}
    }
    assert pres == posts and all(call for _, call in pres)
    assert len(pres) == sum(r["event_name"] == "PreToolUse" for r in acks)
    assert any(
        r["event_name"] == "PostToolUseFailure" and r["call"] == "fixture-eio"
        for r in acks
    )
    return observed


def qualify(output=None, profile="production"):
    out = (
        Path(output)
        if output
        else Path(__file__).parent / "evidence" / ("native-" + str(time.time_ns()))
    )
    out.mkdir(parents=True, exist_ok=False)
    stack = None
    sid, child = "dojo-native-qualification", "dojo-native-qualification-child"
    results = {
        "scope": "native callback and environment-boundary fixtures; no model or official task scores",
        "inference_calls": 0,
        "provider_cost_usd": 0,
        "folder": str(out),
    }
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("native_bridge.mjs"),
        Path(__file__).with_name("stack.py"),
    ]
    results["source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
    }
    try:
        stack = NativeStack(out / "native", profile=profile)
        stack.start(sid)
        stack.request(
            "prompt",
            session=sid,
            text="Run isolated native-stack qualification; no model inference.",
        )

        def call(name, args):
            return body(stack.tool(sid, name, args))

        results["catalog"] = call("memory_status", {"view": "tools"})
        assert len(stack.catalog) == 19
        results["initial_health"] = call("memory_status", {"view": "health"})
        assert results["initial_health"]["totalMemories"] == 0, (
            "prior trial memories entered the fresh store"
        )
        precursor = call(
            "smart_ingest",
            {
                "content": "Verified isolated precursor for service DojoNativeQualification.",
                "node_type": "fact",
                "tags": ["dojo-native-qualification"],
                "entities": ["service:DojoNativeQualification"],
            },
        )
        call("recall", {"handle": precursor["nodeId"]})
        rsb_before = call("memory", {"action": "get", "id": precursor["nodeId"]})
        trigger = call(
            "smart_ingest",
            {
                "content": "FAILURE: service DojoNativeQualification failed in an isolated qualification fixture.",
                "node_type": "event",
                "tags": ["dojo-native-qualification", "failure"],
                "entities": ["service:DojoNativeQualification"],
                "links": [{"kind": "derived_from", "to": precursor["nodeId"]}],
            },
        )
        call("backfill", {"trigger_id": trigger["nodeId"], "scope": "user"})
        receipts = call("backfill", {"action": "receipts"})
        salient = [
            r
            for r in receipts.get("receipts", [])
            if r.get("trigger_id") == trigger["nodeId"]
            and r.get("memory_id") == precursor["nodeId"]
        ]
        assert salient, "eligible native RSB fixture must produce a salience receipt"
        rsb_after = call("memory", {"action": "get", "id": precursor["nodeId"]})
        assert rsb_after["node"]["stability"] > rsb_before["node"]["stability"]
        assert salient[-1]["old_score"] == rsb_before["node"]["stability"]
        assert salient[-1]["new_score"] == rsb_after["node"]["stability"]
        rsb_proofs = [
            call("receipt", {"action": "get", "receipt_id": r["receipt_id"]})
            for r in salient
        ]
        for proof in rsb_proofs:
            verify_receipt(proof)
        results["rsb"] = {
            "precursor": precursor["nodeId"],
            "trigger": trigger["nodeId"],
            "receipts": salient,
            "before": rsb_before,
            "after": rsb_after,
            "effect_receipts": rsb_proofs,
            "claim_boundary": "Native recorded-edge or shared-identity relevance feedback; no causal efficacy inference",
        }

        nodes = []
        for i in range(5):
            args = {
                "content": f"Isolated FSRS qualification fact number {i} for service DojoReview{i}.",
                "node_type": "fact",
                "tags": ["dojo-fsrs-qualification"],
                "entities": [f"service:DojoReview{i}"],
            }
            if nodes:
                args["links"] = [{"kind": "derived_from", "to": nodes[-1]}]
            nodes.append(call("smart_ingest", args)["nodeId"])
        fsrs = []
        for action, node in zip(["promote", "demote"], nodes):
            before = call("memory", {"action": "get", "id": node})
            review = call(
                "memory",
                {
                    "action": action,
                    "id": node,
                    "reason": "isolated native FSRS qualification",
                },
            )
            after = call("memory", {"action": "get", "id": node})
            rid = review.get("receiptId")
            assert rid, "native review must return its effect identity"
            receipt = call("receipt", {"action": "get", "receipt_id": rid})
            verify_receipt(receipt)
            verify_review(before, review, after, action)
            fsrs.append(
                {
                    "action": action,
                    "id": node,
                    "before": before,
                    "review": review,
                    "after": after,
                    "receipt": receipt,
                }
            )
        results["fsrs_reviews"] = fsrs
        dream = call("maintain", {"action": "dream", "memory_count": 5})
        results["dream"] = dream
        assert (
            dream.get("basis") == "recorded_edges_fsrs"
            and dream.get("memoriesReviewed", 0) > 0
        )
        reviews = dream["reviews"]
        assert len(reviews) == dream["memoriesReviewed"]
        assert len({r["id"] for r in reviews}) == len(reviews)
        results["dream_receipts"] = []
        results["dream_states"] = []
        for review in reviews:
            receipt = call("receipt", {"action": "get", "receipt_id": review["id"]})
            verify_receipt(receipt)
            assert any(
                m["id"] == review["id"] and m["kind"] == "promoted"
                for m in receipt["receipt"]["mutations"]
            )
            state = call("memory", {"action": "get", "id": review["id"]})
            assert state["node"]["reps"] == review["repsBefore"] + 1
            results["dream_receipts"].append(receipt)
            results["dream_states"].append(state)

        fixture = stack.root / "task-files" / "fixture.txt"
        fixture.write_text("native benign marker\n")
        allowed = stack.request(
            "before",
            session=sid,
            call="fixture-read",
            tool="Read",
            args={"file_path": str(fixture)},
        )
        assert allowed["allowed"]
        value = fixture.read_text()
        stack.request(
            "after",
            session=sid,
            call="fixture-read",
            tool="Read",
            args={"file_path": str(fixture)},
            result=value,
        )
        # Do not repeat an action the owner gate previously stopped, including
        # earlier classification fixtures. Denied-effect evidence is retained
        # in its original proof; this run proves the current benign boundary.
        results["operator_boundary"] = {
            "benign": allowed,
            "denial_fixture": "not repeated: prior owner stop preserved",
            "dangerous_command_executed": False,
        }
        structured_args = {"query": "isolated harmless fixture"}
        structured = stack.request(
            "before",
            session=sid,
            call="fixture-structured",
            tool="search_documents",
            args=structured_args,
        )
        assert structured["allowed"]
        assert (
            structured["receipt"]["tool"] == "mcp__benchmark-custom__search_documents"
        )
        stack.request(
            "after",
            session=sid,
            call="fixture-structured",
            tool="search_documents",
            args=structured_args,
            result=[],
        )
        results["structured_boundary"] = structured
        failed_args = {"file_path": str(fixture)}
        assert stack.request(
            "before", session=sid, call="fixture-eio", tool="Read", args=failed_args
        )["allowed"]
        stack.request(
            "after",
            session=sid,
            call="fixture-eio",
            tool="Read",
            args=failed_args,
            result=None,
            error="EIO: isolated callback qualification failure",
        )

        stack.start(child, parent=sid)
        body(stack.tool(child, "recall", {"handle": precursor["nodeId"]}))
        stack.stop(child, parent=sid)
        results["compaction"] = stack.request(
            "compact",
            session=sid,
            summary="Synthetic callback qualification summary; intentionally omits precursor handles. No live model compaction is claimed.",
        )
        stack.stop(sid)
        ledger = stack.request("ledger")
        results["ledger"] = ledger
        assert ledger["chain"]["status"] == "ok" and ledger["chain"]["breaks"] == 0
        stack.close()
        rows = [
            json.loads(line)
            for f in (stack.root / "audit").glob("*.jsonl")
            for line in f.read_text().splitlines()
        ]
        acks = [r for r in rows if r.get("event") == "reflex_ack"]
        observed = verify_hooks(acks, sid, child)
        operator_rows = [
            json.loads(line)
            for path in (stack.root / "operator" / "receipts").glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        operator_sessions = sorted({row.get("session", "") for row in operator_rows})
        assert set(operator_sessions) == {sid, child}, (
            "unrelated Operator session history entered the trial"
        )
        results["operator_history_isolation"] = {
            "sessions": operator_sessions,
            "receipt_rows": len(operator_rows),
            "initial_memories": results["initial_health"]["totalMemories"],
        }
        identities = stack.root / "native-identities.json"
        results["native_identities"] = {
            "path": str(identities),
            "sha256": hashlib.sha256(identities.read_bytes()).hexdigest(),
        }
        assert results["source_sha256"] == {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
        }
        results.update(
            status="PASS", hooks=sorted(observed), native_ack_count=len(acks)
        )
        (out / "proof.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "folder": str(out),
                    "hooks": sorted(observed),
                    "tools": 19,
                    "rsb_receipts": len(salient),
                    "fsrs_reviews": len(fsrs),
                    "inference_calls": 0,
                }
            )
        )
        return results
    except BaseException as error:
        if stack is not None:
            stack.abort()
        (out / "FAILED.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "partial_results": results,
                },
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=["production", "all_rules"], default="production"
    )
    qualify(profile=parser.parse_args().profile)
