"""Unscored, zero-provider proof of native memory reachability on official LME.

The lookup is deliberately independent of the question and gold annotations:
read the exact ``history`` tag and follow only native pagination. Official
answer-session annotations are opened after retrieval, solely to check that
returned native records include the relevant evidence. This establishes
reachability and provenance, not semantic ranking or answer accuracy. Nothing
from this proof is forwarded to a scored solver.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import uuid

from .datasets import SOURCES, load_longmemeval, sha256
from .runtime import atomic_write, canonical, read_jsonl, utc_now

SEED = 20261101
SCHEMA = "native-recall-proof-v1"
INITIAL_RECALL = {"handle": "history", "detail_level": "full", "limit": 100}


class RecallProofError(RuntimeError):
    pass


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixed_ids(items):
    ids = [item.id for item in items]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise RecallProofError("proof requires the complete 500-item official manifest")
    return sorted(random.Random(SEED).sample(ids, 10))


def collect_recall_pages(call, *, max_pages):
    """Read exact native pages without a gold-dependent handle or cursor."""
    arguments = deepcopy(INITIAL_RECALL)
    pages, nodes, cursors, ids = [], [], set(), set()
    for _ in range(max_pages):
        cursor = canonical(arguments)
        if cursor in cursors:
            raise RecallProofError("native recall pagination repeated a cursor")
        cursors.add(cursor)
        result = call("recall", arguments)
        if not isinstance(result, dict) or result.get("error") or not isinstance(result.get("nodes"), list):
            raise RecallProofError("native recall did not return memory nodes")
        if result.get("handle") != "history" or result.get("kind") != "tag":
            raise RecallProofError("native recall did not resolve the exact history tag")
        for node in result["nodes"]:
            if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not isinstance(node.get("content"), str):
                raise RecallProofError("native full-detail memory lacks id/content")
            if node["id"] in ids:
                raise RecallProofError("native recall pagination duplicated a memory")
            ids.add(node["id"])
            nodes.append(node)
        pages.append({"arguments": deepcopy(arguments), "result": result})
        following = result.get("next")
        if following is None:
            if not nodes:
                raise RecallProofError("native recall returned an empty memory set")
            return pages, nodes
        if (not isinstance(following, dict) or following.get("handle") != "history"
                or following.get("detail_level") != "full"
                or type(following.get("offset")) is not int
                or following["offset"] != len(nodes)):
            raise RecallProofError("native pagination changed the lookup or skipped records")
        arguments = deepcopy(following)
    raise RecallProofError("native recall pagination did not terminate within the source bound")


def native_pairs(events):
    """Require matched native request/reply evidence; returned errors stay raw."""
    pending, pairs = None, []
    for line, event in enumerate(events, 1):
        if event.get("kind") == "tool_request":
            if pending is not None:
                raise RecallProofError("native request has no serialized response")
            pending = (line, event)
        elif event.get("kind") in {"tool_response", "tool_error"}:
            if pending is None or pending[1].get("name") != event.get("name"):
                raise RecallProofError("native reply does not match its request")
            if event.get("kind") == "tool_error":
                raise RecallProofError("native proof contains a transport/tool error")
            pairs.append((pending[0], pending[1], line, event))
            pending = None
    if pending is not None:
        raise RecallProofError("native proof ends in an unanswered request")
    return pairs


def source_bindings(history, pairs, nodes):
    """Bind returned node bytes to this item's actual normal-path admissions."""
    expected = []
    for index, text in enumerate(history):
        expected.append({"content": text, "node_type": "note",
                         "tags": ["history", f"session-{index}"], "source": f"history:{index}"})
    observed, bindings = [], {}
    for request_line, request, response_line, response in pairs:
        if request.get("name") != "smart_ingest":
            continue
        args, result = request["arguments"], response.get("result")
        if "items" in args and set(args) != {"items"}:
            raise RecallProofError("batch admission contains a privileged or substituted argument")
        batch = args.get("items") if "items" in args else [args]
        replies = result.get("results") if "items" in args and isinstance(result, dict) else [result]
        if not isinstance(batch, list) or not isinstance(replies, list) or len(batch) != len(replies):
            raise RecallProofError("native admission reply does not cover its raw input batch")
        for record, admission in zip(batch, replies, strict=True):
            observed.append(record)
            if not isinstance(admission, dict) or not isinstance(admission.get("nodeId"), str):
                raise RecallProofError("a normal-path source record lacks a real admitted node id")
            identity, content = admission["nodeId"], record.get("content")
            if identity in bindings and bindings[identity]["content"] != content:
                raise RecallProofError("one admitted node was substituted for different raw source bytes")
            bindings[identity] = {"content": content, "content_sha256": text_hash(content),
                "request_line": request_line, "response_line": response_line}
    if observed != expected:
        raise RecallProofError("native writes differ from the exact gold-free official history")
    for node in nodes:
        bound = bindings.get(node.get("id"))
        if bound is None or node.get("content") != bound["content"]:
            raise RecallProofError("retrieved node is not this item's admitted source record")
    if {node["id"] for node in nodes} != set(bindings):
        raise RecallProofError("full-tag recall omitted an admitted source memory")
    return bindings


def audit_relevance(raw, nodes):
    """Gold-only offline audit. This function never invokes an engine/model."""
    session_ids, sessions, dates = raw["haystack_session_ids"], raw["haystack_sessions"], raw["haystack_dates"]
    targets = raw.get("answer_session_ids")
    if not isinstance(targets, list) or not targets:
        raise RecallProofError("official item has no answer-session relevance annotation")
    if len(session_ids) != len(sessions) or len(dates) != len(sessions):
        raise RecallProofError("official relevance annotations are not aligned")
    returned = {node["content"]: node for node in nodes}
    matches = []
    for target in targets:
        positions = [index for index, session in enumerate(session_ids) if session == target]
        if len(positions) != 1:
            raise RecallProofError("official answer-session id is missing or ambiguous")
        index = positions[0]
        evidence_turns = [turn for turn in sessions[index] if turn.get("has_answer") is True]
        # The session is the official evidence unit when turn-level labels are
        # absent. The audit still requires all of its original turns returned.
        turns = evidence_turns or sessions[index]
        if not turns:
            raise RecallProofError("official answer session has no evidence turns")
        for turn in turns:
            text = f'[{dates[index]}] {turn["role"]}: {turn["content"]}'
            if text not in returned:
                raise RecallProofError("native recall omitted an official relevant evidence turn")
            node = returned[text]
            matches.append({"answer_session_id": target, "node_id": node["id"],
                "content_sha256": text_hash(text), "content": text,
                "basis": "official has_answer turn" if evidence_turns else "official answer-session membership"})
    return matches


def _audit_rows(runs):
    source = SOURCES["longmemeval"]
    path = Path(runs) / "upstream/longmemeval" / source["data_file"]
    if sha256(path) != source["data_sha256"]:
        raise RecallProofError("official LongMemEval source hash changed")
    return {str(row["question_id"]): row for row in json.loads(path.read_text())}


def require_independent_stores(entries):
    """Process ids may be reused by the OS; independent data paths may not."""
    try:
        roots = [str(Path(entry["root"]).resolve()) for entry in entries]
        stores = [str(Path(entry["engine"]["data_dir"]).resolve()) for entry in entries]
    except (KeyError, TypeError) as exc:
        raise RecallProofError("native retrieval proof lacks independent store identities") from exc
    if len(set(roots)) != len(roots) or len(set(stores)) != len(stores):
        raise RecallProofError("native retrieval proof reused an item root or data directory")


def _run_item(repo, runs, item, config_hash, directory):
    from .engine import HookedRealEngine, VestigeMCPError
    from .native_contract import model_tools
    root = Path(directory)
    engine = HookedRealEngine(repo, root / "native", manifest_path=Path(runs) / "build-manifest.json",
                              hook_profile="memory_qa")
    record = {"item_id": item.id, "question": item.question, "config_hash": config_hash,
              "started_at": utc_now(), "status": "PENDING"}
    try:
        engine.start()
        record["engine"] = deepcopy(engine.identity)
        # An independent second tools/list request observes the live child.
        catalog = engine.client.request("tools/list")
        atomic_write(root / "catalog-check.json", canonical(catalog) + "\n")
        if catalog.get("tools") != engine.tools or engine.model_tools() != model_tools(catalog["tools"], engine.initialize_result):
            raise RecallProofError("model catalog/schema differs from the actual native tools/list")
        engine.ingest_history(item.history)
        engine.begin_turn(item.question)
        pages, nodes = collect_recall_pages(engine.call, max_pages=len(item.history) + 1)
        atomic_write(root / "recall-pages.json", canonical(pages) + "\n")
        missing_handle = "__vestige-proof-absent-" + uuid.uuid4().hex
        missing = engine.call("recall", {"handle": missing_handle, "detail_level": "full"})
        if missing.get("nodes") or missing.get("error") != "handle_required":
            raise RecallProofError("nonexistent exact-handle negative control returned memories")
        pid = engine.client._proc.pid
        engine.client._proc.kill()
        returncode = engine.client._proc.wait(timeout=10)
        engine._log({"kind": "proof_midrun_kill", "pid": pid, "returncode": returncode})
        try:
            engine.call("recall", deepcopy(INITIAL_RECALL))
        except VestigeMCPError as exc:
            engine._log({"kind": "proof_runner_aborted", "pid": pid,
                         "exception": type(exc).__name__ + ": " + str(exc), "fallback_used": False})
        else:
            raise RecallProofError("dead native child did not abort recall")
        record.update(status="PASS", negative_handle=missing_handle,
                      returned_nodes=len(nodes), recall_pages=len(pages))
    except Exception as exc:
        record.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
    finally:
        if engine.client is not None:
            atomic_write(root / "startup-stderr.log", "\n".join(engine.client.stderr_tail(10000)) + "\n")
        engine.close()
    record.update(finished_at=utc_now(), root=str(root.resolve()),
                  native_log=str((root / "native/mcp.jsonl").resolve()))
    paths = [root / "native/mcp.jsonl", root / "native/mcp-initialize.json", root / "native/mcp-tools.json",
             root / "catalog-check.json", root / "recall-pages.json", root / "native/hooks-runtime/hooks.jsonl"]
    record["artifact_hashes"] = {str(path.resolve()): sha256(path) for path in paths if path.is_file()}
    atomic_write(root / "item.json", canonical(record) + "\n")
    return record


def _validate_item(repo, runs, entry, item, raw, config_hash, binary_hash):
    from .native_contract import contract_identity, model_tools
    if entry.get("status") != "PASS" or entry.get("config_hash") != config_hash:
        raise RecallProofError("retrieval proof item failed or claims another configuration")
    root = Path(entry["root"]).resolve()
    if not root.is_relative_to(Path(runs).resolve()) or root.is_symlink():
        raise RecallProofError("retrieval proof directory escapes the campaign")
    for name, digest in entry["artifact_hashes"].items():
        path = Path(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root) or sha256(path) != digest:
            raise RecallProofError("native retrieval evidence changed or escapes its item")
    native = root / "native"
    required = {str(path) for path in [native / "mcp.jsonl", native / "mcp-initialize.json",
        native / "mcp-tools.json", root / "catalog-check.json", root / "recall-pages.json",
        native / "hooks-runtime/hooks.jsonl"]}
    if not required.issubset(entry["artifact_hashes"]):
        raise RecallProofError("retrieval proof omits required raw evidence hashes")
    initialized = json.loads((native / "mcp-initialize.json").read_text())
    catalog = json.loads((native / "mcp-tools.json").read_text())["tools"]
    if json.loads((root / "catalog-check.json").read_text()).get("tools") != catalog:
        raise RecallProofError("live tools/list differs from startup catalog")
    events = read_jsonl(native / "mcp.jsonl")
    starts = [event for event in events if event.get("kind") == "engine_start"]
    if len(starts) != 1:
        raise RecallProofError("proof lacks exactly one native engine start")
    start = starts[0]
    identity = start["identity"]
    if (start.get("tools") != catalog or start.get("initialize") != initialized
            or start.get("model_tools") != model_tools(catalog, initialized)
            or identity.get("native_contract") != contract_identity(catalog, initialized)):
        raise RecallProofError("native startup/model contract evidence differs")
    if (identity.get("binary_sha256") != binary_hash or identity.get("fresh_store_nodes") != 0
            or identity.get("transport") != "stdio-jsonrpc"
            or identity.get("client") != "benchmarks/memconflict/mcp_client.VestigeMCP"
            or identity.get("data_dir") != str(native / "strata")
            or type(identity.get("pid")) is not int or identity["pid"] <= 0
            or entry.get("engine") != identity
            or entry.get("native_log") != str(native / "mcp.jsonl")):
        raise RecallProofError("proof child binary, transport, or fresh isolated data directory differs")
    pairs = native_pairs(events)
    pages = json.loads((root / "recall-pages.json").read_text())
    history_pairs = [(line, request, reply_line, reply) for line, request, reply_line, reply in pairs
                     if request.get("name") == "recall" and request.get("arguments", {}).get("handle") == "history"]
    if len(history_pairs) != len(pages):
        raise RecallProofError("saved recall pages differ from actual native calls")
    replay = iter(history_pairs)
    def observed_call(name, arguments):
        _, request, _, reply = next(replay)
        if name != request["name"] or arguments != request["arguments"]:
            raise RecallProofError("lookup arguments differ from native pagination")
        return reply["result"]
    actual_pages, nodes = collect_recall_pages(observed_call, max_pages=len(item.history) + 1)
    if pages != actual_pages:
        raise RecallProofError("saved memory contents differ from actual native replies")
    if entry.get("returned_nodes") != len(nodes) or entry.get("recall_pages") != len(pages):
        raise RecallProofError("reported native retrieval counts differ from raw evidence")
    bindings = source_bindings(item.history, pairs, nodes)
    negatives = [reply["result"] for _, request, _, reply in pairs
                 if request.get("name") == "recall" and request.get("arguments", {}).get("handle") == entry.get("negative_handle")]
    if (not str(entry.get("negative_handle", "")).startswith("__vestige-proof-absent-")
            or len(negatives) != 1 or negatives[0].get("nodes")
            or negatives[0].get("error") != "handle_required"):
        raise RecallProofError("proof lacks an actual nonexistent-handle negative control")
    kills = [event for event in events if event.get("kind") == "proof_midrun_kill"]
    aborts = [event for event in events if event.get("kind") == "proof_runner_aborted"]
    if (len(kills) != 1 or len(aborts) != 1 or kills[0].get("pid") != identity.get("pid")
            or aborts[0].get("pid") != identity.get("pid") or kills[0].get("returncode") == 0
            or aborts[0].get("fallback_used") is not False or not aborts[0].get("exception")):
        raise RecallProofError("proof lacks the actual same-child kill/abort control")
    # Relevance labels enter only here, after all raw native requests/replies
    # and their source-byte bindings were verified above.
    relevant = audit_relevance(raw, nodes)
    if entry.get("relevant_memories") != relevant:
        raise RecallProofError("reported relevant memories differ from independent offline audit")
    return {"item_id": item.id, "returned_nodes": len(nodes), "admitted_nodes": len(bindings),
            "relevant_memories": len(relevant), "native_calls": len(pairs), "tool_count": len(catalog)}


def validate_recall_proof(repo, runs, plan, official_items):
    """Independently re-read raw receipts; never trust proof PASS flags alone."""
    repo, runs = Path(repo).resolve(), Path(runs).resolve()
    specification = plan.get("native_retrieval_proof") or {}
    path = Path(specification.get("path", ""))
    if not path.is_absolute() or not path.is_relative_to(runs) or not path.is_file():
        raise RecallProofError("native retrieval proof artifact is absent or outside campaign")
    if sha256(path) != specification.get("sha256"):
        raise RecallProofError("native retrieval proof hash differs from its frozen plan")
    proof = json.loads(path.read_text())
    expected = fixed_ids(official_items)
    binary_hash = json.loads((runs / "build-manifest.json").read_text())["binary_sha256"]
    if (proof.get("schema") != SCHEMA or proof.get("config_hash") != plan.get("config_hash")
            or proof.get("binary_sha256") != binary_hash or proof.get("seed") != SEED
            or proof.get("item_ids") != expected or proof.get("dataset_sha256") != SOURCES["longmemeval"]["data_sha256"]
            or proof.get("status") != "PASS"):
        raise RecallProofError("native retrieval proof is failed, stale, or not the fixed ten official items")
    by_id = {item.id: item for item in official_items}
    entries = proof.get("items", [])
    if len(entries) != 10 or sorted(entry.get("item_id") for entry in entries) != expected:
        raise RecallProofError("native retrieval proof item coverage is missing or duplicated")
    require_independent_stores(entries)
    raw = _audit_rows(runs)
    results = [_validate_item(repo, runs, entry, by_id[entry["item_id"]], raw[entry["item_id"]],
                              plan["config_hash"], binary_hash) for entry in entries]
    return {"status": "PASS", "item_ids": expected, "items": results, "proof_path": str(path),
            "sha256": specification["sha256"], "claim": "native full-tag reachability, not semantic ranking"}


async def run_recall_proof(repo, runs, items, config_hash, workers=10):
    """Create ten fresh native stores; performs no model/provider requests."""
    repo, runs = Path(repo).resolve(), Path(runs).resolve()
    selected = fixed_ids(items)
    by_id = {item.id: item for item in items}
    run_root = runs / "longmemeval/recall-proof" / (utc_now().replace(":", "").replace("+", "-") + "-" + uuid.uuid4().hex[:8])
    run_root.mkdir(parents=True, exist_ok=False)
    semaphore = asyncio.Semaphore(max(1, min(workers, 10)))
    async def execute(identity):
        async with semaphore:
            return await asyncio.to_thread(_run_item, repo, runs, by_id[identity], config_hash,
                                           run_root / text_hash(identity)[:20])
    entries = await asyncio.gather(*(execute(identity) for identity in selected))
    # Open official relevance labels only once all retrieval calls have ended.
    audit_rows = _audit_rows(runs)
    for entry in entries:
        if entry["status"] == "PASS":
            try:
                pages = json.loads((Path(entry["root"]) / "recall-pages.json").read_text())
                nodes = [node for page in pages for node in page["result"]["nodes"]]
                events = read_jsonl(Path(entry["native_log"]))
                source_bindings(by_id[entry["item_id"]].history, native_pairs(events), nodes)
                entry["relevant_memories"] = audit_relevance(audit_rows[entry["item_id"]], nodes)
            except Exception as exc:
                entry.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
        atomic_write(Path(entry["root"]) / "item.json", canonical(entry) + "\n")
    proof = {"schema": SCHEMA, "created_at": utc_now(), "config_hash": config_hash,
        "binary_sha256": json.loads((runs / "build-manifest.json").read_text())["binary_sha256"],
        "dataset_sha256": SOURCES["longmemeval"]["data_sha256"], "seed": SEED, "item_ids": selected,
        "status": "PASS" if all(entry["status"] == "PASS" for entry in entries) else "FAIL",
        "claim": "full-tag native reachability and offline relevant-evidence coverage; not semantic ranking or answer accuracy",
        "paid_model_calls": 0, "items": entries}
    path = runs / "longmemeval/recall-proof.json"
    atomic_write(path, canonical(proof) + "\n")
    if proof["status"] == "PASS":
        validate_recall_proof(repo, runs, {"config_hash": config_hash,
            "native_retrieval_proof": {"path": str(path), "sha256": sha256(path)}}, items)
    report = ["# LongMemEval-S native recall proof", "", f"Status: **{proof['status']}**. Created: {proof['created_at']}",
        "", "Ten fixed-seed official items; fresh native child and data directory per item; zero model/API requests.",
        "Generic exact history-tag reads follow native pagination. Official relevance labels are used only after retrieval and are never engine/model/hook inputs.",
        "This proves native memory reachability and source binding. It does not establish semantic query ranking or benchmark answer accuracy.",
        "", "| Item | Status | Returned memories | Relevant evidence turns | Raw native log |", "|---|---|---:|---:|---|"]
    for entry in entries:
        report.append(f"| {entry['item_id']} | {entry['status']} | {entry.get('returned_nodes', 0)} | {len(entry.get('relevant_memories', []))} | {entry['native_log']} |")
        if entry.get("error"):
            report.append("\nFailure: " + entry["error"] + "\n")
    for entry in [e for e in entries if e["status"] == "PASS"][:2]:
        report.extend(["", "## Raw example: " + entry["item_id"], "", "Question: " + entry["question"],
                       "", "Lookup: `" + canonical(INITIAL_RECALL) + "` followed by the native next cursors."])
        for memory in entry["relevant_memories"][:2]:
            report.extend(["", "Native node: `" + memory["node_id"] + "`; content SHA-256: `" + memory["content_sha256"] + "`.",
                           "", "```text", memory["content"], "```", "", "Offline relevance basis: " + memory["basis"] + "."])
    atomic_write(runs / "longmemeval/recall-proof.md", "\n".join(report) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    repo, runs = args.repo.resolve(), args.repo.resolve() / "runs"
    config = json.loads((runs / "four-config.json").read_text())["config_hash"]
    proof = asyncio.run(run_recall_proof(repo, runs, load_longmemeval(runs / "upstream/longmemeval"), config, args.workers))
    print(canonical({"status": proof["status"], "item_ids": proof["item_ids"], "paid_model_calls": 0}))
    if proof["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
