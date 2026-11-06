"""Explicit, evidence-preserving reuse of unchanged LongMemEval baselines.

This module never runs a provider, a retriever, an engine, or a scorer. A source
response is eligible only when its scientific inputs are unchanged and its
actual request/response usage reconciles to the existing shared spend ledger.
Operational changes (worker count and API concurrency) do not relabel an old
response as a new inference: every imported row retains an authenticated origin.
Vestige rows are deliberately ineligible; the owner requested all 500 rerun.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import fcntl
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3

from .datasets import SOURCES
from .runtime import canonical, record_key, shard_id, _money

BASELINE_ARMS = ("bare", "bm25", "embedding")
SCHEMA = "longmemeval.baseline_resume.v1"
BASELINE_SOURCE_PINS = (
    "benchmarks/real_engine/retrieval.py",
    "benchmarks/real_engine/hook_model.py",
    "benchmarks/real_engine/datasets.py",
)
OFFICIAL_DATA_SHA256 = SOURCES["longmemeval"]["data_sha256"]
OFFICIAL_SCORER_SHA256 = SOURCES["longmemeval"]["scorer_sha256"]


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@lru_cache(maxsize=4096)
def _file_digest_at_state(path, state):
    value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if _file_state(Path(path)) != state:
        raise ValueError("resume evidence changed while being hashed")
    return value


def _file_state(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def file_digest(path):
    path = str(Path(path).resolve())
    return _file_digest_at_state(path, _file_state(path))


def _inside(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("resume evidence escapes its preserved source directory")
    return path.resolve()


@lru_cache(maxsize=4096)
def _records_at_digest(path, expected_digest):
    path = Path(path)
    with path.open("rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("resume evidence changed while being read")
    result = []
    for number, line in enumerate(raw.split(b"\n"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("resume evidence row must be an object")
        result.append((number, row, hashlib.sha256(line).hexdigest()))
    return tuple(result)


def _records(path):
    """Read-only physical/canonical rows cached against a checked file digest."""
    path = str(Path(path).resolve())
    return _records_at_digest(path, file_digest(path))


def _config_hash(config):
    calculated = digest({k: v for k, v in config.items() if k != "config_hash"})
    if config.get("config_hash") != calculated:
        raise ValueError("resume config content differs from its claimed hash")
    return calculated


def _item(item, name):
    return item[name] if isinstance(item, dict) else getattr(item, name)


def _tools(arm):
    # The current baseline tool definition is compared byte-for-byte with the
    # actual source requests. A changed tool contract cannot reuse an old run.
    from .run import SEARCH_TOOL
    return [] if arm == "bare" else [SEARCH_TOOL]


def _prompt(config, arm):
    prompt = config["prompts"][arm]
    if hashlib.sha256(prompt["text"].encode()).hexdigest() != prompt["sha256"]:
        raise ValueError("baseline prompt text/hash mismatch")
    return prompt


def _scientific_signature(repo, old, current, old_plan, current_plan,
                          old_identity, current_identity, official_items, arm):
    if arm not in BASELINE_ARMS:
        raise ValueError("Vestige and non-baseline arms cannot be imported")
    _config_hash(old)
    _config_hash(current)
    for config, plan, identity in ((old, old_plan, old_identity),
                                   (current, current_plan, current_identity)):
        if (config.get("model") != "gpt-5.4-nano" or config.get("temperature") != 0
                or plan.get("config_hash") != config["config_hash"]
                or identity.get("config_hash") != config["config_hash"]
                or identity.get("benchmark") != "longmemeval"
                or identity.get("model") != config["model"] or identity.get("trials") != 1):
            raise ValueError("baseline model, trial, benchmark or config identity changed")
        if config.get("limits") != [6, 2048, 32768] or identity.get("limits") != config["limits"]:
            raise ValueError("baseline aggregate limits changed")
        if plan.get("budgets", {}).get(arm) != {"max_calls": 6, "max_output": 2048, "context": 32768}:
            raise ValueError("baseline declared budget changed")
        if plan.get("run_identities", {}).get(arm) not in (
                digest({**identity, "arm": arm}), [digest({**identity, "arm": arm})]):
            raise ValueError("baseline checkpoint identity is not bound to its inputs")
        if identity.get("prompts", {}).get(arm) != _prompt(config, arm):
            raise ValueError("baseline input identity has a different prompt")
    if _prompt(old, arm) != _prompt(current, arm):
        raise ValueError("baseline prompt changed")
    if not old.get("episode_policy") or old["episode_policy"] != current.get("episode_policy"):
        raise ValueError("baseline episode termination policy changed")
    for field in ("items", "retrieval"):
        if not old_identity.get(field) or old_identity[field] != current_identity.get(field):
            raise ValueError("baseline question/history or retrieval protocol changed")
    ids = [_item(item, "id") for item in official_items]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError("resume requires all 500 unique official LongMemEval-S IDs")
    for plan in (old_plan, current_plan):
        if plan.get("full_set", {}).get("item_ids") != ids or plan["full_set"].get("trials") != [0]:
            raise ValueError("resume plan is not the unchanged full official set")
    pins = {}
    for relative in BASELINE_SOURCE_PINS:
        expected = old.get("sources", {}).get(relative)
        if (not expected or current.get("sources", {}).get(relative) != expected
                or file_digest(_inside(Path(repo)/relative, repo)) != expected):
            raise ValueError("baseline scientific source changed: " + relative)
        pins[relative] = expected
    old_files, new_files = old_plan["full_set"].get("dataset_files"), current_plan["full_set"].get("dataset_files")
    if (not isinstance(old_files, dict) or len(old_files) != 1 or old_files != new_files
            or set(old_files.values()) != {OFFICIAL_DATA_SHA256}):
        raise ValueError("official dataset pin changed")
    for path, sha in old_files.items():
        if file_digest(_inside(path, repo)) != sha:
            raise ValueError("official dataset bytes differ from their pin")
    if old_plan.get("scorer") != current_plan.get("scorer"):
        raise ValueError("official scorer pin changed")
    scorer = old_plan.get("scorer", {})
    files = scorer.get("source_files", {})
    if (scorer.get("origin") != "official" or scorer.get("entrypoint") != "judge_longmemeval"
            or len(files) != 1 or set(files.values()) != {OFFICIAL_SCORER_SHA256}):
        raise ValueError("resume requires the pinned official LongMemEval scorer")
    for path, sha in files.items():
        if file_digest(_inside(path, repo)) != sha:
            raise ValueError("official scorer bytes differ from their pin")
    return {"model": current["model"], "temperature": 0, "limits": current["limits"],
            "prompt": _prompt(current, arm), "baseline_tools": _tools(arm), "source_pins": pins,
            "dataset_files": old_files, "scorer": scorer,
            "items": old_identity["items"], "retrieval": old_identity["retrieval"],
            "episode_policy": old["episode_policy"], "trials": [0]}


def _verify_message_extension(prior_messages, provider_choice, following, arm, item):
    """Only the actual assistant reply and exact official baseline records append.

    Provider-authenticated usage alone cannot excuse an extra user/system hint
    in a later request. No source prompt, response or tool output is repaired.
    """
    message = provider_choice["message"]
    assistant = {k: v for k, v in message.items() if k in ("role", "content", "tool_calls")}
    prefix = prior_messages + [assistant]
    if following[:len(prefix)] != prefix:
        raise ValueError("source appended message differs from actual provider response")
    calls = [] if provider_choice.get("finish_reason") == "length" else message.get("tool_calls") or []
    tail = following[len(prefix):]
    if len(tail) != len(calls):
        raise ValueError("source later prompt has extra hints or missing tool output")
    history = _item(item, "history")
    for call, output in zip(calls, tail, strict=True):
        function = call.get("function", {})
        if (arm == "bare" or function.get("name") != "memory_search"
                or set(output) != {"role", "tool_call_id", "content"}
                or output.get("role") != "tool" or output.get("tool_call_id") != call.get("id")):
            raise ValueError("source baseline tool response is unmatched")
        args = json.loads(function["arguments"])
        hits = json.loads(output["content"])
        if (set(args) - {"query", "k"} or not isinstance(args.get("query"), str)
                or type(args.get("k", 5)) is not int or not isinstance(hits, list)
                or len(hits) > min(10, max(1, args.get("k", 5)))
                or output["content"] != canonical(hits)):
            raise ValueError("source baseline tool contract/content mismatch")
        seen = set()
        for hit in hits:
            if not isinstance(hit, dict) or set(hit) != {"id", "content", "score"}:
                raise ValueError("source baseline tool output has unknown content")
            match = re.fullmatch(r"session-(\d+)", str(hit["id"]))
            index = int(match.group(1)) if match else -1
            score = hit["score"]
            if (index < 0 or index >= len(history) or index in seen or hit["content"] != history[index]
                    or type(score) not in (int, float) or not math.isfinite(score)
                    or (arm == "embedding" and not -1.000001 <= score <= 1.000001)):
                raise ValueError("source tool output is not an unchanged official history record")
            seen.add(index)


def _verify_api(row, path, item, config, ledger_path):
    from .provider import account
    requests, responses = {}, {}
    for number, event, _ in _records(path):
        kind = event.get("event")
        if kind not in {"request", "response"}:
            continue
        call_id = event.get("call_id")
        bucket = requests if kind == "request" else responses
        if not call_id or call_id in bucket:
            raise ValueError("duplicate or unidentified source provider event")
        bucket[call_id] = (number, event)
    if not requests or set(requests) != set(responses) or len(requests) > 6:
        raise ValueError("completed baseline lacks matched source request/response evidence")
    usage_rows, costs, final_message = [], [], None
    previous_messages, previous_choice = None, None
    first = True
    with sqlite3.connect(f"file:{Path(ledger_path).resolve()}?mode=ro", uri=True) as ledger:
        ledger.row_factory = sqlite3.Row
        for call_id, (_, request) in requests.items():
            reply_line, reply = responses[call_id]
            if reply_line <= requests[call_id][0]:
                raise ValueError("provider response precedes its request")
            for entry in (request, reply):
                ident = entry.get("identity", {})
                for key in ("benchmark", "arm", "item_id", "trial", "config_hash", "phase"):
                    if ident.get(key) != row.get(key):
                        raise ValueError("source provider identity mismatch")
            payload = request.get("payload", {})
            expected_fields = {"model", "messages", "max_completion_tokens", "reasoning_effort", "temperature", "store"}
            if row["arm"] != "bare":
                expected_fields |= {"tools", "parallel_tool_calls"}
            if (payload.get("model") != "gpt-5.4-nano" or payload.get("temperature") != 0
                    or payload.get("reasoning_effort") != "none" or payload.get("store") is not False
                    or set(payload) != expected_fields
                    or payload.get("max_completion_tokens") != min(512, 2048-sum(u["completion_tokens"] for u in usage_rows))
                    or payload.get("tools", []) != _tools(row["arm"])
                    or (row["arm"] != "bare" and payload.get("parallel_tool_calls") is not False)):
                raise ValueError("source model, generation budget or tool contract differs")
            messages = payload.get("messages", [])
            expected = [{"role": "system", "content": _prompt(config, row["arm"])["text"]},
                        {"role": "user", "content": _item(item, "question")}]
            if messages[:2] != expected or (first and (messages != expected or row.get("prompt") != expected)):
                raise ValueError("source actual prompt differs from this official item")
            if previous_choice is not None:
                if (previous_choice.get("finish_reason") == "length"
                        or not previous_choice["message"].get("tool_calls")):
                    raise ValueError("source made another call after its completed baseline episode")
                _verify_message_extension(previous_messages, previous_choice, messages, row["arm"], item)
            first = False
            import tiktoken
            actual_context = len(tiktoken.get_encoding("o200k_base").encode(canonical(
                {"messages": messages, "tools": payload.get("tools", [])})))
            if actual_context > 32768:
                raise ValueError("source request exceeds the unchanged context cap")
            response = reply.get("response", {})
            if (response.get("model") != "gpt-5.4-nano"
                    and not str(response.get("model", "")).startswith("gpt-5.4-nano-")):
                raise ValueError("source actual provider model is not the required nano model")
            cost, usage = account(response)
            if (usage != reply.get("usage") or _money(cost) != _money(reply.get("cost"))
                    or any(type(usage.get(k)) is not int for k in ("prompt_tokens", "completion_tokens"))
                    or usage["prompt_tokens"] <= 0 or usage["completion_tokens"] < 0
                    or usage["completion_tokens"] > payload["max_completion_tokens"]):
                raise ValueError("source provider actual usage/cost mismatch")
            attempts = [call_id + f"/attempt-{attempt}" for attempt in range(7)]
            matches = ledger.execute("SELECT actual,usage FROM calls WHERE id IN (?,?,?,?,?,?,?) AND state='settled'", attempts).fetchall()
            if (len(matches) != 1 or matches[0]["actual"] != _money(cost)
                    or json.loads(matches[0]["usage"]) != usage):
                raise ValueError("source usage is not authenticated to one settled shared-ledger attempt")
            usage_rows.append(usage)
            costs.append(Decimal(str(cost)))
            choices = response.get("choices", [])
            if not choices or not isinstance(choices[0].get("message"), dict):
                raise ValueError("source provider response has no actual message")
            final_message = choices[0]["message"]
            previous_messages, previous_choice = messages, choices[0]
    _verify_message_extension(previous_messages, previous_choice, row.get("transcript", []), row["arm"], item)
    expected_answer = ((final_message.get("content") or "")
        if previous_choice.get("finish_reason") == "length" or not final_message.get("tool_calls") else "")
    if (row.get("usage") != usage_rows or sum(u["completion_tokens"] for u in usage_rows) > 2048
            or abs(Decimal(str(row.get("cost"))) - sum(costs)) > Decimal("0.000000000001")
            or row.get("model") != response.get("model") or row.get("response") != expected_answer):
        raise ValueError("source checkpoint response, usage or cost differs from actual provider evidence")


def _source_row(record, manifest, repo, official_items, ledger_path):
    path = _inside(record["source_checkpoint"], manifest["source_root"])
    matches = [(r, raw_sha) for n, r, raw_sha in _records(path) if n == record["source_line"]]
    if len(matches) != 1:
        raise ValueError("source checkpoint row is missing")
    row, raw_sha = matches[0]
    if raw_sha != record["source_line_sha256"] or digest(row) != record["source_row_sha256"]:
        raise ValueError("source checkpoint content changed")
    if (row.get("origin") or row.get("arm") not in BASELINE_ARMS
            or row.get("benchmark") != "longmemeval" or row.get("status") != "complete"
            or row.get("score") is not None or row.get("trial") != 0
            or row.get("config_hash") != manifest["source_config"]["config_hash"]
            or row.get("run_identity") != digest({**manifest["source_identity"], "arm": row["arm"]})
            or row.get("budget", {}).get("limits") != {"max_calls": 6, "max_output": 2048, "context": 32768}):
        raise ValueError("source checkpoint is not an original compatible unscored baseline")
    for field, source_field in (("arm", "arm"), ("item_id", "item_id"), ("trial", "trial"),
                                ("config_hash", "source_config_hash"), ("run_identity", "source_run_identity"),
                                ("worker_id", "source_worker_id"), ("shard", "source_shard"), ("raw_log", "raw_log")):
        if row.get(field) != record.get(source_field):
            raise ValueError("source origin metadata differs from its checkpoint")
    old_workers = manifest["source_config"].get("workers")
    if (type(old_workers) is not int or old_workers <= 0
            or row["worker_id"] != shard_id(row["item_id"], workers=old_workers)
            or row["shard"] != row["worker_id"] or path.name != f"worker-{row['worker_id']}.jsonl"
            or record["target_worker_id"] != shard_id(row["item_id"], workers=30)):
        raise ValueError("source or destination shard is not deterministic")
    item_map = {_item(item, "id"): item for item in official_items}
    if row["item_id"] not in item_map:
        raise ValueError("source checkpoint item is not in the official set")
    raw_log = _inside(record["resolved_raw_log"], manifest["source_root"])
    if file_digest(raw_log) != record["raw_log_sha256"]:
        raise ValueError("source API evidence changed")
    _verify_api(row, raw_log, item_map[row["item_id"]], manifest["source_config"], ledger_path)
    return deepcopy(row), raw_log


def prepare_baseline_resume(*, repo, source_root, source_config_path, source_plan,
                            source_identity, current_config, current_plan,
                            current_identity, official_items, workers=30,
                            arms=BASELINE_ARMS, source_original_root=None, ledger_path=None):
    """Read/validate source evidence and return a manifest; do not mutate it.

    Freeze the returned manifest and put its file path/SHA in the current
    qualification plan under ``baseline_resume`` before importing any row.
    ``source_original_root`` identifies pre-archive raw_log paths, when moved.
    """
    repo, source_root = Path(repo).resolve(), Path(source_root).resolve()
    config_path = _inside(source_config_path, repo)
    source_config = json.loads(config_path.read_text())
    if workers != 30 or not arms or set(arms) - set(BASELINE_ARMS):
        raise ValueError("only 30-worker baseline recovery is authorized")
    if not source_root.is_relative_to(repo):
        raise ValueError("preserved baseline source directory escapes the repository")
    ledger_path = Path(ledger_path or repo/"runs/budget.db").resolve()
    original_root = Path(source_original_root or repo/"runs/longmemeval").resolve()
    signatures = {arm: _scientific_signature(repo, source_config, current_config, source_plan,
                    current_plan, source_identity, current_identity, official_items, arm) for arm in arms}
    manifest = {"schema": SCHEMA, "source_root": str(source_root),
        "source_config_file": {"path": str(config_path), "sha256": file_digest(config_path)},
        "source_config": source_config, "source_plan": deepcopy(source_plan),
        "source_identity": deepcopy(source_identity), "current_config_hash": current_config["config_hash"],
        "current_run_identities": {a: digest({**current_identity, "arm": a}) for a in arms},
        "scientific_signature": signatures, "workers": workers, "records": []}
    seen = set()
    for arm in arms:
        for path in sorted((source_root/arm).glob("worker-*.jsonl")):
            if not re.fullmatch(r"worker-\d+\.jsonl", path.name):
                continue
            for number, row, raw_sha in _records(path):
                key = (arm, row.get("item_id"), row.get("trial"))
                if key in seen:
                    raise ValueError("duplicate source item/trial across baseline shards")
                seen.add(key)
                raw_log = Path(row.get("raw_log", ""))
                original_path = raw_log if raw_log.is_absolute() else repo/raw_log
                try:
                    resolved_raw = source_root/original_path.resolve().relative_to(original_root)
                except ValueError:
                    raise ValueError("source raw_log cannot be traced to its original benchmark root")
                record = {"arm": arm, "item_id": row.get("item_id"), "trial": row.get("trial"),
                    "source_checkpoint": str(path.resolve()), "source_line": number,
                    "source_line_sha256": raw_sha, "source_row_sha256": digest(row),
                    "source_config_hash": row.get("config_hash"), "source_run_identity": row.get("run_identity"),
                    "source_worker_id": row.get("worker_id"), "source_shard": row.get("shard"),
                    "raw_log": row.get("raw_log"), "resolved_raw_log": str(resolved_raw.resolve()),
                    "raw_log_sha256": file_digest(_inside(resolved_raw, source_root)),
                    "target_worker_id": shard_id(row.get("item_id"), workers=workers)}
                _source_row(record, manifest, repo, official_items, ledger_path)
                manifest["records"].append(record)
    manifest["counts"] = {arm: sum(r["arm"] == arm for r in manifest["records"]) for arm in arms}
    official_ids = [_item(item, "id") for item in official_items]
    manifest["expected_items_per_arm"] = 500
    manifest["target_shards"] = {identity: shard_id(identity, workers=30) for identity in official_ids}
    manifest["remaining_item_ids"] = {arm: sorted(set(official_ids) - {
        r["item_id"] for r in manifest["records"] if r["arm"] == arm}) for arm in arms}
    return manifest


def validate_imported_baseline(row, manifest, repo, current_config, current_plan,
                               current_identity, official_items, ledger_path=None):
    """Return authenticated original evidence; never waive it as current data."""
    if manifest.get("schema") != SCHEMA or row.get("arm") not in BASELINE_ARMS:
        raise ValueError("unknown resume schema or prohibited Vestige import")
    origin = row.get("origin", {})
    if origin.get("manifest_sha256") != digest(manifest):
        raise ValueError("import origin is not bound to the exact resume manifest")
    record = {k: v for k, v in origin.items() if k != "manifest_sha256"}
    matches = [r for r in manifest.get("records", []) if r == record]
    if len(matches) != 1:
        raise ValueError("import source origin is missing or duplicated")
    if file_digest(manifest["source_config_file"]["path"]) != manifest["source_config_file"]["sha256"]:
        raise ValueError("frozen source config changed")
    if json.loads(Path(manifest["source_config_file"]["path"]).read_text()) != manifest["source_config"]:
        raise ValueError("embedded source config differs from its frozen file")
    signature = _scientific_signature(repo, manifest["source_config"], current_config,
        manifest["source_plan"], current_plan, manifest["source_identity"], current_identity,
        official_items, row["arm"])
    if (signature != manifest["scientific_signature"].get(row["arm"])
            or manifest["current_config_hash"] != current_config["config_hash"]
            or manifest["workers"] != 30):
        raise ValueError("resume scientific signature or current identity changed")
    source, raw_log = _source_row(record, manifest, repo, official_items,
                                 ledger_path or Path(repo)/"runs/budget.db")
    transformed = deepcopy(source)
    transformed.update(config_hash=current_config["config_hash"],
        run_identity=digest({**current_identity, "arm": row["arm"]}),
        worker_id=record["target_worker_id"], shard=record["target_worker_id"],
        raw_log=str(raw_log), origin={**record, "manifest_sha256": digest(manifest)},
        source_run_identity=source["run_identity"])
    if row != transformed:
        raise ValueError("import changed response, prompt, usage or other source evidence")
    pilot = manifest["source_plan"].get("pilot", {})
    frozen_ids = sorted(random.Random(20261101).sample([_item(i, "id") for i in official_items], 10))
    source_pilot = (pilot.get("seed") == 20261101 and pilot.get("item_ids") == frozen_ids
                    and pilot.get("config_hash") == source["config_hash"]
                    and source["item_id"] in frozen_ids and source.get("phase") == "integrity_preflight")
    return {"source_row": source, "raw_log_path": str(raw_log), "record": record,
            "source_pilot": source_pilot}


def import_baseline_resume(manifest, checkpoints, *, repo, current_config, current_plan,
                           current_identity, official_items, ledger_path=None):
    """Append only verified original baselines to their new operational shards."""
    counts = {arm: 0 for arm in manifest["counts"]}
    for record in manifest["records"]:
        arm = record["arm"]
        if arm not in BASELINE_ARMS:
            raise ValueError("Vestige imports are prohibited")
        cp = checkpoints[arm][record["target_worker_id"]]
        if cp.identity != manifest["current_run_identities"][arm]:
            raise ValueError("destination checkpoint identity differs from resume target")
        key = record_key(record)
        if key in cp.done:
            continue
        source, raw_log = _source_row(record, manifest, repo, official_items,
                                     ledger_path or Path(repo)/"runs/budget.db")
        row = deepcopy(source)
        row.update(config_hash=current_config["config_hash"], run_identity=cp.identity,
            worker_id=record["target_worker_id"], shard=record["target_worker_id"],
            raw_log=str(raw_log), origin={**record, "manifest_sha256": digest(manifest)},
            source_run_identity=source["run_identity"])
        validate_imported_baseline(row, manifest, repo, current_config, current_plan,
                                   current_identity, official_items, ledger_path)
        cp.append(row)
        counts[arm] += 1
    return counts
