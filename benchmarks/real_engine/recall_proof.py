"""Unscored proof of native memory reachability and all active hooks on LME.

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
import sqlite3
import uuid

from .datasets import SOURCES, load_longmemeval, sha256
from .runtime import atomic_write, canonical, read_jsonl, shard_id, utc_now

SEED = 20261101
SCHEMA = "native-recall-proof-v2-all-hooks"
INITIAL_RECALL = {"handle": "history", "detail_level": "full", "limit": 100}
ITEM_LIMITS = {"max_calls": 6, "max_output": 2048, "context": 32768}


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


def forwarded_hook_errors(root):
    """Exact raw errors visible in actual later hook model requests only."""
    visible = []
    for row in read_jsonl(Path(root) / "hook-api.jsonl"):
        if row.get("event") != "request":
            continue
        pending = {}
        for message in row.get("payload", {}).get("messages", []):
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    pending[call.get("id")] = call.get("function", {})
            elif message.get("role") == "tool":
                call = pending.get(message.get("tool_call_id"), {})
                try:
                    arguments = json.loads(call["arguments"])
                    result = json.loads(message["content"])
                except (KeyError, TypeError, ValueError):
                    continue
                if isinstance(result, dict) and result.get("isError") is True:
                    visible.append({"name": str(call.get("name", "")).removeprefix("mcp__vestige__"),
                        "arguments": arguments, "result": result, "seen_at": row.get("timestamp")})
    return visible


def native_pairs(events, forwarded_errors=()):
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
                raw = event.get("raw_mcp_result")
                def moment(value):
                    from datetime import datetime
                    if isinstance(value, (int, float)):
                        return float(value)
                    try:
                        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                    except (ValueError, TypeError, AttributeError):
                        return None
                when = moment(event.get("timestamp"))
                matched = any(visible.get("name") == pending[1].get("name")
                    and visible.get("arguments") == pending[1].get("arguments")
                    and visible.get("result") == raw and when is not None
                    and moment(visible.get("seen_at")) is not None
                    and when <= moment(visible["seen_at"]) for visible in forwarded_errors)
                if not isinstance(raw, dict) or raw.get("isError") is not True or not matched:
                    raise RecallProofError("native proof contains an unforwarded native error or transport failure")
                event = {**event, "result": raw, "native_error_forwarded_unchanged": True}
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
            if not isinstance(admission, dict):
                raise RecallProofError("a normal-path source record lacks a native admission")
            content = record.get("content")
            if admission.get("success") is True and admission.get("decision") == "reinforce":
                # Native smart_ingest records a duplicate echo, not a new source
                # node. Bind both ids to this item's earlier admission and the
                # exact native canonical hash; never treat the echo as raw text.
                original, identity = admission.get("duplicateOf"), admission.get("echoNodeId")
                target = bindings.get(original)
                digest = admission.get("canonicalHash")
                if (not isinstance(identity, str) or not identity or target is None
                        or not isinstance(digest, str) or len(digest) != 64
                        or target.get("canonical_hash") != digest
                        or not isinstance(admission.get("receiptId"), str)):
                    raise RecallProofError("reinforced admission lacks same-item native target/hash/echo provenance")
                line_count = len(content.split("\n")) - int(content.endswith("\n")) if content else 0
                echo = f"duplicate of {original}\ncanonical_hash: {digest}\nsubmitted_line_count: {line_count}"
                bound = {"content": echo, "content_sha256": text_hash(echo),
                    "request_line": request_line, "response_line": response_line,
                    "duplicate_of": original, "echo_node_id": identity,
                    "submitted_content_sha256": text_hash(content), "canonical_hash": digest,
                    "receipt_id": admission["receiptId"]}
            else:
                identity = admission.get("nodeId")
                if not isinstance(identity, str):
                    raise RecallProofError("a normal-path source record lacks a real admitted node id")
                bound = {"content": content, "content_sha256": text_hash(content),
                    "request_line": request_line, "response_line": response_line,
                    "canonical_hash": admission.get("canonicalHash")}
            if identity in bindings and bindings[identity]["content"] != bound["content"]:
                raise RecallProofError("one admitted node was substituted for different raw source bytes")
            bindings[identity] = bound
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


def audit_hook_models(root, runs, item_id, config_hash, *, verify_ledger=True):
    """Reconcile every proof-hook response to actual Nano usage and ledger rows."""
    from .provider import MODEL, account
    from .runtime import _money
    root, runs = Path(root), Path(runs)
    api = read_jsonl(root / "hook-api.jsonl")
    requests = [row for row in api if row.get("event") == "request"]
    responses = [row for row in api if row.get("event") == "response"]
    attempts = [row for row in api if row.get("event") == "attempt"]
    solver = read_jsonl(root / "api.jsonl")
    if any(row.get("event") == "request" for row in solver):
        raise RecallProofError("native proof unexpectedly issued a solver request")
    if len(requests) != len(responses) or len(requests) > ITEM_LIMITS["max_calls"]:
        raise RecallProofError("hook proof has an incomplete or over-budget model request")
    by_id = {row.get("call_id"): row for row in requests}
    if len(by_id) != len(requests) or None in by_id:
        raise RecallProofError("hook proof duplicates or omits model call identities")
    if len({row.get("call_id") for row in responses}) != len(responses):
        raise RecallProofError("hook proof duplicates a model response")
    usage, cost, output = [], 0.0, 0
    unresolved = 0
    ledger = sqlite3.connect(f"file:{runs / 'budget.db'}?mode=ro", uri=True) if requests and verify_ledger else None
    if ledger is not None:
        ledger.row_factory = sqlite3.Row
    try:
        for response in responses:
            request = by_id.get(response.get("call_id"))
            if request is None:
                raise RecallProofError("hook response has no original request")
            for row in (request, response):
                identity = row.get("identity", {})
                if (identity.get("item_id") != item_id or identity.get("config_hash") != config_hash
                        or identity.get("benchmark") != "longmemeval" or identity.get("arm") != "vestige"
                        or identity.get("phase") != "hook"):
                    raise RecallProofError("hook model request/response belongs to another item or role")
            payload = request.get("payload", {})
            if payload.get("model") != MODEL or payload.get("temperature") != 0:
                raise RecallProofError("proof hook requested a different model/temperature")
            import tiktoken
            context_tokens = len(tiktoken.get_encoding("o200k_base").encode(canonical({
                "messages": payload.get("messages", []), "tools": payload.get("tools", [])})))
            if context_tokens > ITEM_LIMITS["context"]:
                raise RecallProofError("proof hook exceeded the shared item context cap")
            cap = payload.get("max_completion_tokens")
            if type(cap) is not int or cap <= 0 or cap > ITEM_LIMITS["max_output"] - output:
                raise RecallProofError("proof hook exceeded the aggregate requested output cap")
            try:
                actual_cost, actual_usage = account(response.get("response", {}))
            except Exception as exc:
                raise RecallProofError("proof hook provider omitted valid actual usage") from exc
            if any(type(actual_usage.get(key)) is not int or actual_usage[key] < 0
                   for key in ("prompt_tokens", "completion_tokens")):
                raise RecallProofError("proof hook usage is not nonnegative integer token counts")
            if (actual_usage != response.get("usage") or _money(actual_cost) != _money(response.get("cost"))
                    or actual_usage["completion_tokens"] > cap):
                raise RecallProofError("proof hook response usage/cost is inconsistent")
            output += actual_usage["completion_tokens"]
            if output > ITEM_LIMITS["max_output"]:
                raise RecallProofError("proof hook exceeded the shared item output budget")
            usage.append(actual_usage)
            cost += actual_cost
            if ledger is not None:
                attempt_ids = [str(response["call_id"]) + f"/attempt-{index}" for index in range(7)]
                rows = ledger.execute("SELECT * FROM calls WHERE id IN (?,?,?,?,?,?,?)", attempt_ids).fetchall()
                settled = [row for row in rows if row["state"] == "settled"]
                if (len(settled) != 1 or settled[0]["actual"] != _money(actual_cost)
                        or json.loads(settled[0]["usage"]) != actual_usage):
                    raise RecallProofError("proof hook usage is not reconciled to one settled shared-ledger attempt")
                unresolved += sum(row["reserved"] for row in rows if row["state"] == "reserved")
        after = [row for row in read_jsonl(root / "budget.jsonl") if row.get("event") == "budget_after"]
        if requests and (len(after) != len(responses) or [row.get("usage") for row in after] != usage
                         or any(row.get("limits") != ITEM_LIMITS for row in after)):
            raise RecallProofError("proof hook response usage differs from the shared per-item budget trace")
    finally:
        if ledger is not None:
            ledger.close()
    return {"model": MODEL, "limits": ITEM_LIMITS, "successful_calls": len(responses),
            "request_count": len(requests), "attempt_count": len(attempts), "usage": usage,
            "cost": cost, "completion_tokens": output, "unresolved_reserved": unresolved / 1e9}


def audit_hook_lifecycle(repo, root, *, profile="all_applicable_hooks"):
    """Verify each source registration fired naturally in this item's client."""
    from .hooks import merge_fragments, read_audit
    root, repo = Path(root).resolve(), Path(repo).resolve()
    hooks_root = root / "native/hooks-runtime"
    events = read_jsonl(hooks_root / "hooks.jsonl")
    prepared = [row for row in events if row.get("kind") == "prepared"]
    expected, wired = merge_fragments(repo, profile)
    if (len(prepared) != 1 or prepared[0].get("hook_profile") != profile
            or prepared[0].get("wired") != wired):
        raise RecallProofError("proof hooks do not match the exact source settings registrations")
    audited = read_audit(repo, repo / "runs/hook-audit.json")
    if prepared[0].get("source_pins") != audited["source_pins"]:
        raise RecallProofError("proof hook source pins differ from the current source audit")
    paths = prepared[0].get("paths", {})
    if (Path(paths.get("home", "/")).resolve() != hooks_root / "home"
            or prepared[0].get("cwd_git_ancestor_check", {}).get("git_ancestors") != []):
        raise RecallProofError("proof hook HOME/cwd is not isolated from ambient memory and Git")
    failed_bridges = [row for row in events if row.get("kind") == "bridge_error"]
    if failed_bridges:
        raise RecallProofError("proof hook transport/model callback failed: " + str(failed_bridges[-1].get("error")))
    outcomes = [row for row in events if row.get("kind") in {"hook_event", "hook_not_applicable"}]
    registrations = []
    for event_name, entries in expected.items():
        for entry in entries:
            hits = [row for row in outcomes if row.get("event") == event_name and row.get("entry") == entry]
            fired = [row for row in hits if row.get("kind") == "hook_event"]
            if not fired:
                raise RecallProofError("registered hook never fired in the actual proof lifecycle: " + entry["command"])
            registrations.append({"event": event_name, "entry": entry, "fired_count": len(fired),
                                  "outcomes": hits})
    for row in outcomes:
        if (row.get("kind") == "hook_not_applicable" and row.get("entry") is None
                and row.get("event") in expected and not expected[row["event"]]
                and isinstance(row.get("reason"), str) and row["reason"].strip()):
            continue  # An actual event with no source registration is not an omitted hook.
        if row.get("entry") not in expected.get(row.get("event"), []):
            raise RecallProofError("unwired or unrelated hook appears in the proof lifecycle")
        if row.get("kind") == "hook_not_applicable":
            if not row.get("reason"):
                raise RecallProofError("unreached hook registration has no explicit reason")
            continue
        if row.get("input", {}).get("cwd") != paths.get("cwd"):
            raise RecallProofError("hook event ran in another item's working directory")
        if row.get("exit_code") not in {0, 2}:
            raise RecallProofError("original hook command failed")
        if row.get("exit_code") == 2 or row.get("blocked"):
            raise RecallProofError("original hook blocked proof: " + str(row.get("blocking_reasons", row.get("stderr", ""))))
        if profile == "all_applicable_hooks" and any(key not in row for key in
                ("injected_context", "blocked", "blocking_reasons")):
            raise RecallProofError("all-hook proof lacks exact injection/block outcome metadata")
    return {"hook_profile": profile, "registered_hooks": registrations, "events": outcomes}


def _run_item(repo, runs, item, config_hash, directory, hook_model_callback=None,
              hook_profile="all_applicable_hooks"):
    from .engine import HookedRealEngine, VestigeMCPError
    from .native_contract import model_tools
    root = Path(directory)
    engine = HookedRealEngine(repo, root / "native", manifest_path=Path(runs) / "build-manifest.json",
                              hook_profile=hook_profile, hook_model_callback=hook_model_callback)
    record = {"item_id": item.id, "question": item.question, "config_hash": config_hash,
              "started_at": utc_now(), "status": "PENDING", "hook_profile": hook_profile,
              "solver_model_calls": 0, "transcript_origin": "deterministic native proof client"}
    stop_reached = False
    transcript = [{"role": "user", "content": item.question}]
    try:
        if hook_profile == "all_applicable_hooks" and hook_model_callback is None:
            raise RecallProofError("all-active-hook proof requires the campaign's metered callback")
        bound = getattr(hook_model_callback, "bound", None)
        if hook_model_callback is not None and (bound is None or bound.limits != ITEM_LIMITS):
            raise RecallProofError("proof hook callback lacks the equal shared item budget")
        engine.start()
        record["engine"] = deepcopy(engine.identity)
        # An independent second tools/list request observes the live child.
        catalog = engine.client.request("tools/list")
        atomic_write(root / "catalog-check.json", canonical(catalog) + "\n")
        if catalog.get("tools") != engine.tools or engine.model_tools() != model_tools(catalog["tools"], engine.initialize_result):
            raise RecallProofError("model catalog/schema differs from the actual native tools/list")
        engine.ingest_history(item.history)
        begin_context = engine.begin_turn(item.question)
        record["begin_turn_injections"] = deepcopy(begin_context)
        transcript.extend({"role": "user", "content": text} for text in begin_context)
        turn_injections = []
        def recall_call(name, arguments):
            output = engine.call(name, arguments)
            call_id = "native-proof-recall-" + str(len(turn_injections))
            transcript.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function", "function": {"name": name, "arguments": canonical(arguments)}}]})
            transcript.append({"role": "tool", "tool_call_id": call_id, "content": canonical(output)})
            context = engine.drain_context()
            turn_injections.append({"name": name, "arguments_sha256": text_hash(canonical(arguments)),
                                    "injected_context": deepcopy(context)})
            transcript.extend({"role": "user", "content": text} for text in context)
            return output
        pages, nodes = collect_recall_pages(recall_call, max_pages=len(item.history) + 1)
        atomic_write(root / "recall-pages.json", canonical(pages) + "\n")
        missing_handle = "__vestige-proof-absent-" + uuid.uuid4().hex
        missing = recall_call("recall", {"handle": missing_handle, "detail_level": "full"})
        if missing.get("nodes") or missing.get("error") != "handle_required":
            raise RecallProofError("nonexistent exact-handle negative control returned memories")
        record["tool_hook_injections"] = turn_injections
        completion = f"Native recall returned {len(nodes)} stored memory records. This completed a memory retrieval check; no answer to the user's question was generated."
        transcript.append({"role": "assistant", "content": completion})
        record["retrieval_completion"] = completion
        # Stop observes the actual generic retrieval transcript, never a gold
        # answer or a fabricated solver response. Its veto is an actual failure.
        stop_reached = True
        stopped = engine.finish_turn(transcript)
        record["stop_hook_outcome"] = deepcopy(stopped)
        if stopped.get("blocked"):
            raise RecallProofError("Stop hook blocked native retrieval proof: " + "\n".join(stopped.get("reasons", [])))
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
        if engine.hooks is not None and not stop_reached and hasattr(engine.hooks, "not_applicable"):
            engine.hooks.not_applicable("Stop", "Proof aborted before Stop: " + record.get("error", "incomplete native lifecycle"))
        if engine.client is not None:
            atomic_write(root / "startup-stderr.log", "\n".join(engine.client.stderr_tail(10000)) + "\n")
        engine.close()
    record.update(finished_at=utc_now(), root=str(root.resolve()),
                  native_log=str((root / "native/mcp.jsonl").resolve()))
    hook_log = root / "native/hooks-runtime/hooks.jsonl"
    record["hook_events"] = [event for event in read_jsonl(hook_log)
                             if event.get("kind") in {"hook_event", "hook_not_applicable"}] if hook_log.exists() else []
    try:
        record["hook_lifecycle"] = audit_hook_lifecycle(repo, root, profile=hook_profile)
    except Exception as exc:
        record["hook_lifecycle_error"] = str(exc)
        if record["status"] == "PASS":
            record.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
    try:
        record["hook_model"] = audit_hook_models(root, Path(runs), item.id, config_hash,
                                                 verify_ledger=True)
        if hook_model_callback is not None:
            bound = hook_model_callback.bound
            if (bound.usage != record["hook_model"]["usage"]
                    or abs(bound.cost - record["hook_model"]["cost"]) > 1e-12):
                raise RecallProofError("hook callback usage differs from actual metered responses")
    except Exception as exc:
        record["hook_model_error"] = str(exc)
        if record["status"] == "PASS":
            record.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
    paths = [root / "native/mcp.jsonl", root / "native/mcp-initialize.json", root / "native/mcp-tools.json",
             root / "catalog-check.json", root / "recall-pages.json", hook_log,
             root / "hook-api.jsonl", root / "budget.jsonl", root / "hook-model.jsonl"]
    record["artifact_hashes"] = {str(path.resolve()): sha256(path) for path in paths if path.is_file()}
    atomic_write(root / "item.json", canonical(record) + "\n")
    return record


def _validate_item(repo, runs, entry, item, raw, config_hash, binary_hash,
                   required_hook_profile="all_applicable_hooks"):
    from .native_contract import contract_identity, model_tools
    if entry.get("status") != "PASS" or entry.get("config_hash") != config_hash:
        raise RecallProofError("retrieval proof item failed or claims another configuration")
    if entry.get("hook_profile") != required_hook_profile or entry.get("solver_model_calls") != 0:
        raise RecallProofError("retrieval proof used the wrong hook profile or a solver call")
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
    pairs = native_pairs(events, forwarded_hook_errors(root))
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
    hook_lifecycle = audit_hook_lifecycle(repo, root, profile=required_hook_profile)
    hook_model = audit_hook_models(root, runs, item.id, config_hash, verify_ledger=True)
    if (entry.get("hook_lifecycle") != hook_lifecycle or entry.get("hook_events") != hook_lifecycle["events"]
            or entry.get("hook_model") != hook_model):
        raise RecallProofError("reported hook events/injections/blocks or metered usage differ from raw evidence")
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
            "relevant_memories": len(relevant), "native_calls": len(pairs), "tool_count": len(catalog),
            "hook_profile": required_hook_profile, "registered_hook_entries": len(hook_lifecycle["registered_hooks"]),
            "hook_model_calls": hook_model["successful_calls"], "hook_model_cost": hook_model["cost"]}


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
            or proof.get("status") != "PASS" or proof.get("hooks_profile") != "all_applicable_hooks"
            or proof.get("solver_model_calls") != 0):
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


async def run_recall_proof(repo, runs, items, config_hash, workers=10,
                           hook_model_callback_factory=None, hook_profile="all_applicable_hooks"):
    """Create ten native stores; hook model calls use only the supplied meter."""
    if hook_profile != "all_applicable_hooks" or hook_model_callback_factory is None:
        raise RecallProofError("official proof requires all active hooks and the campaign's metered callback factory")
    repo, runs = Path(repo).resolve(), Path(runs).resolve()
    selected = fixed_ids(items)
    by_id = {item.id: item for item in items}
    run_root = runs / "longmemeval/recall-proof" / (utc_now().replace(":", "").replace("+", "-") + "-" + uuid.uuid4().hex[:8])
    run_root.mkdir(parents=True, exist_ok=False)
    semaphore = asyncio.Semaphore(max(1, min(workers, 10)))
    async def execute(identity):
        async with semaphore:
            root = run_root / text_hash(identity)[:20]
            meta = {"benchmark": "longmemeval", "arm": "vestige", "item_id": identity,
                "config_hash": config_hash, "phase": "native_retrieval_proof", "trial": 0,
                "seed": 0, "selection_seed": SEED, "worker_id": shard_id(identity, 30),
                "shard": shard_id(identity, 30), "engine_data_dir": str(root / "native/strata")}
            callback = hook_model_callback_factory(meta, root)
            return await asyncio.to_thread(_run_item, repo, runs, by_id[identity], config_hash,
                                           root, callback, hook_profile)
    entries = await asyncio.gather(*(execute(identity) for identity in selected))
    # Open official relevance labels only once all retrieval calls have ended.
    audit_rows = _audit_rows(runs)
    for entry in entries:
        if entry["status"] == "PASS":
            try:
                pages = json.loads((Path(entry["root"]) / "recall-pages.json").read_text())
                nodes = [node for page in pages for node in page["result"]["nodes"]]
                events = read_jsonl(Path(entry["native_log"]))
                source_bindings(by_id[entry["item_id"]].history,
                                native_pairs(events, forwarded_hook_errors(Path(entry["root"]))), nodes)
                entry["relevant_memories"] = audit_relevance(audit_rows[entry["item_id"]], nodes)
            except Exception as exc:
                entry.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
        atomic_write(Path(entry["root"]) / "item.json", canonical(entry) + "\n")
    proof = {"schema": SCHEMA, "created_at": utc_now(), "config_hash": config_hash,
        "binary_sha256": json.loads((runs / "build-manifest.json").read_text())["binary_sha256"],
        "dataset_sha256": SOURCES["longmemeval"]["data_sha256"], "seed": SEED, "item_ids": selected,
        "status": "PASS" if all(entry["status"] == "PASS" for entry in entries) else "FAIL",
        "claim": "full-tag native reachability and offline relevant-evidence coverage; not semantic ranking or answer accuracy",
        "hooks_profile": hook_profile, "solver_model_calls": 0,
        "hook_model_calls": sum(entry.get("hook_model", {}).get("successful_calls", 0) for entry in entries),
        "hook_model_cost": sum(entry.get("hook_model", {}).get("cost", 0.0) for entry in entries),
        "hook_unresolved_reserved": sum(entry.get("hook_model", {}).get("unresolved_reserved", 0.0) for entry in entries),
        "items": entries}
    path = runs / "longmemeval/recall-proof.json"
    atomic_write(path, canonical(proof) + "\n")
    if proof["status"] == "PASS":
        validate_recall_proof(repo, runs, {"config_hash": config_hash,
            "native_retrieval_proof": {"path": str(path), "sha256": sha256(path)}}, items)
    report = ["# LongMemEval-S native recall proof", "", f"Status: **{proof['status']}**. Created: {proof['created_at']}",
        "", "Ten fixed-seed official items; fresh native child and data directory per item; zero solver model requests.",
        f"All source-registered hooks enabled. Metered hook model calls: {proof['hook_model_calls']}; actual cost: ${proof['hook_model_cost']:.6f}; unresolved reservations: ${proof['hook_unresolved_reserved']:.6f}.",
        "Generic exact history-tag reads follow native pagination. Official relevance labels are used only after retrieval and are never engine/model/hook inputs.",
        "This proves native memory reachability and source binding. It does not establish semantic query ranking or benchmark answer accuracy.",
        "", "| Item | Status | Returned memories | Relevant evidence turns | Raw native log |", "|---|---|---:|---:|---|"]
    for entry in entries:
        report.append(f"| {entry['item_id']} | {entry['status']} | {entry.get('returned_nodes', 0)} | {len(entry.get('relevant_memories', []))} | {entry['native_log']} |")
        if entry.get("error"):
            report.append("\nFailure: " + entry["error"] + "\n")
        report.extend(["", "Hook outcomes for `" + entry["item_id"] + "`:", "", "```json",
                       json.dumps(entry.get("hook_events", []), indent=2, ensure_ascii=False), "```"])
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
    raise SystemExit("Launch the all-active-hook proof through Campaign with its shared metered hook callback; standalone unmetered proof execution is disabled.")


if __name__ == "__main__":
    main()
