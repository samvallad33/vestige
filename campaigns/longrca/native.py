"""Inject an isolated, receipt-verifying native binding; never emulate causal_walk.

The official trace schema carries no typed causal edges. We ingest zero links.
Temporal adjacency, handoff-looking names, and reference labels are not evidence
of a recorded dependency. Kernel-created exact-identity edges remain native
output and must be reported as such rather than called annotated causal truth.
"""

from __future__ import annotations

import copy
from typing import Any, Protocol

from .schema import PublicTrajectory, canonical, validate_public


class NativeBinding(Protocol):
    """Implementation must bind an isolated store, Operator, Reflex and receipts.

    verify_receipt must reverify the native receipt, not trust a response flag.
    store_id must identify a dedicated per-arm/per-trajectory isolated store.
    This module has no client or native binary launch defaults.
    """

    store_id: str

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]: ...

    def verify_receipt(
        self, response: dict[str, Any], tool: str, arguments: dict[str, Any]
    ) -> bool: ...


class NativeTraceSession:
    def __init__(self, public: PublicTrajectory, binding: NativeBinding):
        # Revalidate even if constructed externally, prohibiting extra nested gold fields.
        self.public = validate_public(public.as_dict())
        if not isinstance(binding.store_id, str) or not binding.store_id.strip():
            raise ValueError("an isolated native store identity is mandatory")
        self.binding = binding
        self.nodes: dict[int, str] = {}
        self.events: list[dict[str, Any]] = []
        self._used = False
        self._halted = False

    def _call(
        self, tool: str, arguments: dict[str, Any], *, origin: str
    ) -> dict[str, Any]:
        if self._halted:
            raise RuntimeError("FAIL_CLOSED_INTERDICTION: session already halted")
        args = copy.deepcopy(arguments)
        result = None
        try:
            result = self.binding.call(tool, args)
            if (
                not isinstance(result, dict)
                or not result
                or result.get("isError")
                or result.get("error")
                or result.get("success") is False
                or result.get("status")
                in {"BLOCKED_BY_OPERATOR", "FAIL_CLOSED_INTERDICTION"}
            ):
                raise RuntimeError(
                    "FAIL_CLOSED_INTERDICTION: empty, blocked or failed native response"
                )
            if not self.binding.verify_receipt(result, tool, arguments):
                raise RuntimeError(
                    "FAIL_CLOSED_INTERDICTION: native receipt did not verify"
                )
        except Exception as exc:
            self._halted = True
            self.events.append(
                {
                    "tool": tool,
                    "origin": origin,
                    "arguments": copy.deepcopy(arguments),
                    "status": "interdicted",
                    "response": copy.deepcopy(result),
                    "error_type": type(exc).__name__,
                    "store_id": self.binding.store_id,
                }
            )
            raise RuntimeError(
                "FAIL_CLOSED_INTERDICTION: native call or receipt verification failed"
            ) from exc
        self.events.append(
            {
                "tool": tool,
                "origin": origin,
                "arguments": copy.deepcopy(arguments),
                "response": copy.deepcopy(result),
                "store_id": self.binding.store_id,
                "status": "acknowledged",
            }
        )
        return result

    def ingest(self) -> dict[int, str]:
        if self._used:
            raise RuntimeError(
                "one-shot ingestion; use a fresh isolated store after failure"
            )
        self._used = True
        for item in self.public.history:
            result = self._call(
                "smart_ingest",
                {
                    "content": canonical(
                        {"question_ID": self.public.question_id, "record": item}
                    ).decode(),
                    "node_type": "event",
                    "tags": ["longrca-trace", self.public.question_id],
                    "links": [],
                    "source": "longrca:" + self.public.evidence_sha256,
                },
                origin="controller_write",
            )
            node = result.get("nodeId")
            if not isinstance(node, str) or not node or node in self.nodes.values():
                self._halted = True
                raise RuntimeError(
                    "FAIL_CLOSED_INTERDICTION: missing/duplicate native node identity"
                )
            self.nodes[item["step"]] = node
        return dict(self.nodes)

    def _step(self, step: int) -> str:
        if (
            type(step) is not int
            or step not in self.nodes
            or len(self.nodes) != len(self.public.history)
        ):
            raise ValueError(
                "only a valid solver-selected step in a fully ingested trace may be read"
            )
        return self.nodes[step]

    def read_step(self, step: int) -> dict[str, Any]:
        return self._call(
            "recall",
            {"handle": self._step(step), "detail_level": "full"},
            origin="model_read",
        )

    def causal_walk(self, step: int) -> dict[str, Any]:
        return self._call(
            "causal_walk",
            {
                "start_points": [{"kind": "logged_write", "node_id": self._step(step)}],
                "promote": False,
            },
            origin="model_read",
        )

    def diagnostic(self) -> dict[str, Any]:
        return {
            "type": "native_walk_diagnostic",
            "question_ID": self.public.question_id,
            "evidence_sha256": self.public.evidence_sha256,
            "store_id": self.binding.store_id,
            "controller_supplied_edges": 0,
            "root_or_role_prediction": None,
            "model_reads": sum(
                e["origin"] == "model_read" and e["status"] == "acknowledged"
                for e in self.events
            ),
            "controller_writes": sum(
                e["origin"] == "controller_write" and e["status"] == "acknowledged"
                for e in self.events
            ),
            "qualification": "native runtime and receipt verifier must be independently qualified",
            "halted": self._halted,
            "events": copy.deepcopy(self.events),
        }
