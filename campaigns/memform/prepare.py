"""Prepare official MemForm cases; no model, credential, or environment loading.

Gold labels and interface metadata remain in a private scoring sidecar. Runtime
packets contain only anonymous observed text, formation instructions and tasks.
This native-memory track preserves official item semantics while replacing the
upstream memory stack; it is not an unmodified LangChain/LlamaIndex score.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from campaigns.memform import upstream_mcq_parser

UPSTREAM_URL = "https://github.com/opanhw/memform-code"
PIN = "0565117042a31a5f78576f89ff6b8a77c393ee09"
INTERFACES = ("P1", "P2", "P3", "P4", "P5")
DATA = "outputs/mcq_item_audit/accepted_cases.jsonl"
FREEFORM_DATA = "outputs/freeform_item_audit/accepted_cases.jsonl"
FORBIDDEN = frozenset(
    {
        "clean_answer",
        "poisoned_answer",
        "clean_policy",
        "poisoned_policy",
        "supports",
        "scope_bucket",
        "provenance",
        "authentic",
        "oracle_memory",
        "interface",
        "attack_type",
        "case_id",
        "family_id",
        "condition",
        "contains_forged_evidence",
        "expected_answer",
    }
)


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def upstream_modules(upstream: Path):
    """Load only reviewed, pinned stdlib-only modules; never runner main/load_env."""
    upstream = upstream.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    if commit != PIN:
        raise ValueError("MemForm upstream commit differs from pinned source")
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=upstream,
        text=True,
    )
    if changed:
        raise ValueError("MemForm upstream has modified tracked source/data")
    for name, module in tuple(sys.modules.items()):
        if name == "memform" or name.startswith("memform."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(upstream):
                raise ValueError("another MemForm installation is already loaded")
    sys.path.insert(0, str(upstream))
    try:
        common = importlib.import_module("memform.common")
        defense = importlib.import_module("memform.defenses.common")
        metrics = importlib.import_module("memform.defenses.metrics")
        # The evaluation release's freeform/__init__.py imports a withheld
        # freeform.build module. Common/defense/metrics have no such dependency.
        for module in (common, defense, metrics):
            if not Path(module.__file__).resolve().is_relative_to(upstream):
                raise ValueError("MemForm module path escaped pinned checkout")
        return common, defense, metrics
    finally:
        sys.path.remove(str(upstream))


def assert_no_labels(value: Any) -> None:
    if isinstance(value, dict):
        if FORBIDDEN.intersection(value):
            raise ValueError("gold/provenance metadata entered runtime packet")
        for item in value.values():
            assert_no_labels(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_labels(item)


def parse_mcq_answer(upstream: str | Path, text: str, labels: list[str]):
    """Use the licensed parser copy; only re.I/IGNORECASE alias style differs."""
    upstream_modules(Path(upstream))
    source = Path(upstream).resolve() / "memform/eval/core.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "parse_final_label_details"
    )
    copied = next(
        node
        for node in ast.parse(Path(upstream_mcq_parser.__file__).read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "parse_final_label_details"
    )

    class RegexFlagStyle(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "re"
                and node.attr == "I"
            ):
                node.attr = "IGNORECASE"
            return self.generic_visit(node)

    if ast.dump(RegexFlagStyle().visit(function), include_attributes=False) != ast.dump(
        RegexFlagStyle().visit(copied), include_attributes=False
    ):
        raise ValueError("local parser differs from pinned upstream function")
    return upstream_mcq_parser.parse_final_label_details(text, labels)


def normalize_case(case: dict[str, Any], common, defense) -> tuple[list[dict], dict]:
    errors = common.validate_compiled_case(case)
    if errors:
        raise ValueError("invalid official case: " + "; ".join(errors))
    if case.get("evaluation", {}).get("type") != "multiple_choice":
        raise ValueError(
            "native MCQ preparation requires multiple_choice; free-form scoring remains separate"
        )
    tasks = case["evaluation"]["trigger_tasks"]
    if not tasks:
        raise ValueError("case has no downstream tasks")
    opaque = "mf-" + digest(case["case_id"])[:24]
    task_inputs = [
        {"task": task["task"], "choices": copy.deepcopy(task["choices"])}
        for task in tasks
    ]
    private = {
        "record_id": opaque,
        "case_id": case["case_id"],
        "interface": case["interface"],
        "family_id": case["family_id"],
        "raw_case_sha256": digest(case),
        "tasks": copy.deepcopy(tasks),
        "slots": {},
    }
    mb = case["memory_building"]
    slots = []
    for condition in ("clean", "poisoned"):
        kind, ids = defense.primary_history_ids(case, condition)
        text = (
            mb[f"{condition}_condition"].get("memory_text")
            if kind == "memory_text"
            else None
        )
        slots.append((condition, ids, text))
    for name, (_, ids) in defense.control_histories(case).items():
        slots.append(("control:" + name, ids, None))
    packets = []
    for slot, ids, direct in slots:
        items = defense.items_from_ids(case, ids)
        if len(items) != len(ids):
            raise ValueError("history references missing observations")
        history = [{"text": item.text} for item in items]
        packet = {
            "record_id": "mf-" + digest([case["case_id"], slot])[:24],
            "context": {
                "decision_variable": case["decision_variable"],
                "domain": case["domain"],
            },
            "observations": history,
            "direct_memory_text": direct,
            "formation_instruction": mb.get("batch_prompt")
            or mb.get("incremental_prompt"),
            "tasks": copy.deepcopy(task_inputs),
        }
        assert_no_labels(packet)
        packets.append(packet)
        private["slots"][packet["record_id"]] = {
            "slot": slot,
            "experience_ids": list(ids),
            "packet_sha256": digest(packet),
            "native_formation": "direct_write"
            if direct is not None
            else "ordered_experience_writes",
        }
    # No-memory baseline receives the same tasks without histories. Its opaque ID
    # is controller bookkeeping; the solver receives context/tasks only.
    packet = {
        "record_id": "mf-" + digest([case["case_id"], "baseline"])[:24],
        "context": {
            "decision_variable": case["decision_variable"],
            "domain": case["domain"],
        },
        "observations": [],
        "direct_memory_text": None,
        "formation_instruction": None,
        "tasks": copy.deepcopy(task_inputs),
    }
    packets.append(packet)
    private["slots"][packet["record_id"]] = {
        "slot": "baseline",
        "experience_ids": [],
        "packet_sha256": digest(packet),
        "native_formation": "none",
    }
    return packets, private


def commit_packet(stack, session: str, packet: dict) -> dict:
    """Explicit native writes only; callers supply an admitted/audited native stack.

    No returned content is added to a solver header. Returned node handles enable
    later model-issued recall. P5 order is preserved. This is the native adapter
    treatment, not a claim that Strata performs an LLM memory updater or HRR unbind.
    """
    assert_no_labels(packet)
    contents = (
        [packet["direct_memory_text"]]
        if packet["direct_memory_text"] is not None
        else [row["text"] for row in packet["observations"]]
    )
    handles, receipts = [], []
    for text in contents:
        response = stack.tool(
            session,
            "smart_ingest",
            {
                "content": text,
                "node_type": "fact",
                "tags": ["memform-track", packet["record_id"]],
            },
        )
        if response.get("isError"):
            raise RuntimeError("native memory admission rejected")
        body = response.get("structuredContent")
        if body is None:
            body = json.loads(
                next(
                    row["text"] for row in response["content"] if row["type"] == "text"
                )
            )
        node = body.get("nodeId")
        if not node:
            raise RuntimeError("native write returned no node identity")
        proof = stack.tool(session, "receipt", {"action": "get", "receipt_id": node})
        if proof.get("isError"):
            raise RuntimeError("native receipt rejected")
        receipt = proof.get("structuredContent")
        if receipt is None:
            receipt = json.loads(
                next(row["text"] for row in proof["content"] if row["type"] == "text")
            )
        verification = receipt.get("attestation", {}).get("verification", {})
        if not all(
            verification.get(key) is True
            for key in ("chainValid", "gateAllowed", "locallyVerified")
        ):
            raise RuntimeError("native state receipt did not verify")
        handles.append(node)
        receipts.append(receipt)
    return {
        "handles": handles,
        "receipts": receipts,
        "solver_header_memory_bodies": [],
        "claim": "native ordered-write adapter; downstream dynamic solver reads still required",
    }


def prepare(
    upstream: str | Path, output: str | Path, *, limit: int | None = None
) -> dict:
    upstream = Path(upstream).resolve()
    common, defense, _ = upstream_modules(upstream)
    all_cases = common.read_jsonl(upstream / DATA)
    counts = Counter(case["interface"] for case in all_cases)
    for case in all_cases:
        errors = common.validate_compiled_case(case)
        if errors:
            raise ValueError("official dataset validation failed: " + case["case_id"])
    if set(counts) != set(INTERFACES):
        raise ValueError("official dataset does not cover all five interfaces")
    if limit is not None and (type(limit) is not int or limit < 5):
        raise ValueError("limit must cover at least one item per interface (>=5)")
    # Stratified deterministic sampling is declared before any target result.
    cases = []
    buckets = {
        kind: [row for row in all_cases if row["interface"] == kind]
        for kind in INTERFACES
    }
    for index in range(max(map(len, buckets.values()))):
        for kind in INTERFACES:
            if index < len(buckets[kind]):
                cases.append(buckets[kind][index])
    cases = cases if limit is None else cases[:limit]
    packets, keys = [], []
    for case in cases:
        one, key = normalize_case(case, common, defense)
        packets.extend(one)
        keys.append(key)
    paths = [
        Path(DATA),
        Path(FREEFORM_DATA),
        Path("LICENSE"),
        Path("README.md"),
        *[p.relative_to(upstream) for p in (upstream / "memform").rglob("*.py")],
    ]
    manifest = {
        "benchmark": "MemForm",
        "source_status": "VERIFIED_PINNED",
        "source_url": UPSTREAM_URL,
        "paper_url": "https://arxiv.org/abs/2610.14431v1",
        "commit": PIN,
        "license": "MIT",
        "upstream": str(upstream),
        "track": "native-memory-adapter-MCQ",
        "state": "PREPARED_NOT_EXECUTED",
        "upstream_inventory": dict(counts),
        "freeform_cases": len(common.read_jsonl(upstream / FREEFORM_DATA)),
        "selected_cases": len(cases),
        "runtime_packets": len(packets),
        "selection": "round-robin P1..P5, source order",
        "sha256_files": {str(p): file_sha(upstream / p) for p in sorted(set(paths))},
        "adapter_sha256_files": {
            p.name: file_sha(p) for p in Path(__file__).parent.glob("*.py")
        },
        "runtime_packets_sha256": digest(packets),
        "answer_key_sha256": digest(keys),
        "model_calls": 0,
        "provider_spend_usd": 0,
        "native_protocol": "explicit ordered writes; dynamic exact-handle reads; fresh stores per arm/condition",
        "upstream_protocol_amendment": "native memory stack replaces upstream consolidation; anonymous text only; scores labeled separate track",
        "scoring": "original MemForm fixed-denominator metrics; MCQ parser is upstream; free-form separately gated",
        "upstream_runner_blocker": "freeform/__init__.py imports withheld freeform.build; core runner import fails. MCQ pure validator/metrics and exact parser AST are isolated without upstream source edits.",
        "claim_boundary": "immutable receipt proves admitted bytes, not semantic truth or poisoning immunity",
    }
    manifest["sha256"] = digest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_new(output / "manifest.json", manifest)
    for name, rows in (("runtime-inputs.jsonl", packets), ("answer-key.jsonl", keys)):
        with (output / name).open("x", encoding="utf-8") as stream:
            os.chmod(output / name, 0o600)
            for row in rows:
                stream.write(canonical(row) + "\n")
    return manifest


def load_prepared(output: str | Path) -> tuple[dict, list[dict], list[dict]]:
    root = Path(output)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = manifest.pop("sha256", None)
    if digest(manifest) != expected:
        raise ValueError("prepared manifest identity changed")
    manifest["sha256"] = expected
    upstream = Path(manifest["upstream"])
    common, _, _ = upstream_modules(upstream)
    if any(
        file_sha(upstream / filename) != sha
        for filename, sha in manifest["sha256_files"].items()
    ):
        raise ValueError("official data/source changed after preparation")
    if any(
        file_sha(Path(__file__).parent / filename) != sha
        for filename, sha in manifest["adapter_sha256_files"].items()
    ):
        raise ValueError("adapter changed after preparation")
    packets = common.read_jsonl(root / "runtime-inputs.jsonl")
    private = common.read_jsonl(root / "answer-key.jsonl")
    if (
        digest(packets) != manifest["runtime_packets_sha256"]
        or digest(private) != manifest["answer_key_sha256"]
    ):
        raise ValueError("runtime packets or scoring sidecar changed")
    for packet in packets:
        assert_no_labels(packet)
    return manifest, packets, private


def score_results(
    upstream: str | Path,
    undefended_rows: list[dict],
    method_rows: list[dict],
    *,
    synthetic_fixture=False,
) -> dict:
    """Original scorer; missing/technical failures withhold official ASR/Acc."""
    common, _, metrics = upstream_modules(Path(upstream))
    corpus = {row["case_id"]: row for row in common.read_jsonl(Path(upstream) / DATA)}
    undefended_rows, method_rows = (
        copy.deepcopy(undefended_rows),
        copy.deepcopy(method_rows),
    )
    for rows, baseline in ((undefended_rows, True), (method_rows, False)):
        for row in rows:
            case = corpus.get(row.get("case_id"))
            if case is None and not synthetic_fixture:
                raise ValueError("result case is not in pinned official corpus")
            if case is not None and row.get("interface") != case["interface"]:
                raise ValueError("result interface differs from pinned case")
            tasks = (
                {task["task_id"]: task for task in case["evaluation"]["trigger_tasks"]}
                if case
                else {}
            )
            for task in row.get("task_results", []):
                gold = tasks.get(task.get("task_id"))
                if case is not None and (
                    gold is None
                    or any(
                        task.get(k) != gold[k]
                        for k in ("clean_answer", "poisoned_answer")
                    )
                ):
                    raise ValueError(
                        "result gold labels differ from pinned official task"
                    )
                conditions = (
                    ("baseline", "learned_clean", "learned_poisoned")
                    if baseline
                    else ("learned_clean", "learned_poisoned")
                )
                for name in conditions:
                    condition = task.get(name)
                    valid = isinstance(condition, dict) and (
                        condition.get("defense_blocked")
                        or condition.get("predicted_answer") is not None
                    )
                    if not valid:
                        if baseline:
                            raise ValueError(
                                "undefended qualification has missing/technical prediction; eligibility cannot be frozen"
                            )
                        task[name] = {
                            "defense_technical_failure": True,
                            "defense_outcome_class": "parse_error",
                        }
    duplicates = []
    base = metrics.index_case_rows(undefended_rows, duplicates=duplicates)
    measured = metrics.index_case_rows(method_rows, duplicates=duplicates)
    if duplicates:
        raise ValueError("duplicate case/task results")
    eligibility = metrics.build_eligibility(base)
    result = metrics.score_on_eligibility(measured, eligibility)
    return {
        "scorer": "upstream memform.defenses.metrics",
        "official_source_commit": PIN,
        "track": "native-memory-adapter-MCQ",
        "score_scope": "synthetic scorer fixture"
        if synthetic_fixture
        else "pinned official cases, amended native-memory track",
        "metrics": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = prepare(args.upstream, args.output, limit=args.limit)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "benchmark",
                    "state",
                    "selected_cases",
                    "runtime_packets",
                    "sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
