"""Shared dynamic native memory route for the prepared benchmark adapters.

No model/client or embedding dependency. Histories are explicit writes, not
prompt-header context. Callers own task/environment/provider execution; Terminal-
Bench's guarded host owns those effects. This adapter mediates memory calls only.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

TOOL_NAMES = frozenset(
    {
        "backfill",
        "causal_walk",
        "codebase",
        "dedup",
        "forgotten_lesson",
        "gather",
        "ghostlink",
        "intention",
        "maintain",
        "memory",
        "memory_status",
        "precedent",
        "project",
        "recall",
        "receipt",
        "selftest",
        "session_start",
        "smart_ingest",
        "suppress",
    }
)
GOLD_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "gold",
        "ground_truth",
        "oracle",
        "root_cause",
        "root_step",
        "responsible_role",
        "has_answer",
        "answer_session_ids",
        "answer_turns",
    }
)


class KernelBoundaryError(RuntimeError):
    pass


def native_body(value):
    if not isinstance(value, dict) or value.get("isError"):
        raise KernelBoundaryError(
            "native tool did not return a successful MCP envelope"
        )
    if "structuredContent" in value:
        result = value["structuredContent"]
    else:
        text = next(
            (
                item.get("text")
                for item in value.get("content", [])
                if item.get("type") == "text"
            ),
            None,
        )
        if not isinstance(text, str) or not text.strip():
            raise KernelBoundaryError("empty native MCP envelope")
        result = json.loads(text)
    if (
        not isinstance(result, dict)
        or not result
        or result.get("error")
        or result.get("success") is False
    ):
        raise KernelBoundaryError("malformed or failed native tool body")
    if result.get("status") in {"BLOCKED_BY_OPERATOR", "FAIL_CLOSED_INTERDICTION"}:
        raise KernelBoundaryError("native tool was interdicted")
    return result


def check_history(record):
    if not isinstance(record, dict) or not record:
        raise KernelBoundaryError("history record must be a nonempty object")

    def check_annotations(value):
        if isinstance(value, dict):
            if GOLD_FIELDS.intersection(value):
                raise KernelBoundaryError(
                    "evaluator fields are forbidden in memory history"
                )
            for child in value.values():
                check_annotations(child)
        elif isinstance(value, list):
            for child in value:
                check_annotations(child)

    check_annotations(record)
    if not any(key in record for key in ("content", "turns", "messages", "text")):
        raise KernelBoundaryError("no source observation payload")
    # Original source utterances can contain arbitrary words. Only structured
    # annotations are excluded; never redact evidence based on answer strings.
    return json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def default_factory(root, *, profile):
    from campaigns.terminal_bench.native_vendor.stack import NativeStack

    return NativeStack(root, profile=profile)


class NativeMemory:
    """One isolated trial. A failed native operation permanently closes it."""

    def __init__(self, root, *, profile="all_rules", factory=default_factory):
        self.root = Path(root).resolve()
        self.store_id = "memory-trial-" + uuid.uuid4().hex
        self.session = self.store_id
        self.profile = profile
        self.failed = False
        self.stack = factory(self.root, profile=profile)
        self.catalog = self.stack.catalog
        self.records = []
        try:
            if {tool["name"] for tool in self.catalog} != TOOL_NAMES:
                raise KernelBoundaryError(
                    "native catalog changed; requalify before continuing"
                )
            self.stack.start(self.session)
        except BaseException:
            self.failed = True
            self.stack.abort()
            raise

    def call(self, tool, arguments):
        if self.failed:
            raise KernelBoundaryError("FAIL_CLOSED_INTERDICTION: trial is held")
        if tool not in TOOL_NAMES:
            self.failed = True
            self.stack.abort()
            raise KernelBoundaryError("unknown native tool")
        try:
            return native_body(self.stack.tool(self.session, tool, arguments))
        except BaseException:
            self.failed = True
            self.stack.abort()
            raise

    def verify_receipt(self, node_id):
        try:
            receipt = self.call("receipt", {"action": "get", "receipt_id": node_id})
            attestation = receipt.get("attestation")
            verification = (
                attestation.get("verification")
                if isinstance(attestation, dict)
                else None
            )
            if not isinstance(verification, dict) or not all(
                verification.get(key) is True
                for key in ("chainValid", "gateAllowed", "locallyVerified")
            ):
                raise KernelBoundaryError("native write attestation failed")
            return receipt
        except BaseException:
            self.failed = True
            self.stack.abort()
            raise

    def stage_record(self, record, *, scope, ordinal):
        # Validate before ANY native write; gold never enters the store.
        content = check_history(record)
        if (
            not isinstance(scope, str)
            or not scope
            or type(ordinal) is not int
            or ordinal < 0
        ):
            raise KernelBoundaryError("neutral scope and source ordinal required")
        result = self.call(
            "smart_ingest",
            {
                "content": content,
                "node_type": "event",
                "tags": ["benchmark-history", scope, "observation-" + str(ordinal)],
            },
        )
        node_id = result.get("nodeId")
        if not isinstance(node_id, str) or not node_id:
            self.failed = True
            self.stack.abort()
            raise KernelBoundaryError("history write did not return an exact handle")
        receipt = self.verify_receipt(node_id)
        self.records.append({"node_id": node_id, "scope": scope, "ordinal": ordinal})
        return {"node_id": node_id, "receipt": receipt}

    def milestone(self, node_id):
        if node_id not in {record["node_id"] for record in self.records}:
            raise KernelBoundaryError("milestone requires a recorded trial node")
        # No manufactured failure/outcome delta. Native eligibility decides
        # whether salience changes; a no-op remains part of the evidence.
        return self.call("backfill", {"trigger_id": node_id, "scope": "user"})

    def close(self):
        if self.failed:
            self.stack.abort()
            raise KernelBoundaryError("cannot certify a held trial")
        try:
            self.stack.stop(self.session)
            ledger = self.stack.request("ledger")
            if (
                ledger.get("chain", {}).get("status") != "ok"
                or ledger["chain"].get("breaks") != 0
            ):
                raise KernelBoundaryError("native ledger does not verify")
            self.stack.close()
            return {
                "store_id": self.store_id,
                "profile": self.profile,
                "tools_loaded": len(self.catalog),
                "records": self.records,
                "ledger": ledger,
                "model_calls": None,
                "model_accounting": "caller-owned; this memory adapter does not measure inference",
            }
        except BaseException:
            self.failed = True
            self.stack.abort()
            raise


class TraceBinding:
    """LongRCA binding: native write attestation plus native read-ack integrity.

    Read acknowledgments are Reflex ledger evidence, not invented Strata effect
    signatures. The native host already binds exact args/call/result identities.
    """

    def __init__(self, memory):
        self.memory, self.store_id = memory, memory.store_id
        self.latest = None

    def call(self, tool, arguments):
        response = self.memory.call(tool, arguments)
        self.latest = json.dumps(
            [tool, arguments, response], sort_keys=True, ensure_ascii=False
        )
        return response

    def verify_receipt(self, response, tool, arguments):
        observed = json.dumps(
            [tool, arguments, response], sort_keys=True, ensure_ascii=False
        )
        if self.latest != observed:
            return False
        if tool == "smart_ingest":
            node_id = response.get("nodeId")
            if not node_id:
                return False
            self.memory.verify_receipt(node_id)
        ledger = self.memory.stack.request("ledger")
        return (
            ledger.get("chain", {}).get("status") == "ok"
            and ledger["chain"].get("breaks") == 0
        )
