"""Zero-inference integration smoke for native memory adapters.

Uses source history only, explicit native writes and scripted exact-handle reads.
These fixtures establish wiring/receipts, not model benchmark performance.
"""

import argparse
import json
from pathlib import Path

from benchctl import create_output, sha, source_identity, write_new
from campaigns.longrca.native import NativeTraceSession
from campaigns.longrca.schema import split_record
from campaigns.memform.prepare import commit_packet
from kernel_adapter import NativeMemory, TraceBinding


def first_row(path):
    with Path(path).open() as stream:
        return json.loads(next(stream))


def string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from string_values(child)


def qualify(prepared, output, *, profile="all_rules"):
    prepared = Path(prepared).resolve()
    output = create_output(output)
    sources = source_identity()
    report = {
        "status": "RUNNING",
        "scope": "native memory wiring fixtures, not model scores",
        "profile": profile,
        "model_calls": 0,
        "provider_spend_usd": "0",
        "sources": sources,
        "cases": [],
    }
    memory = None
    try:
        for name in ["memoryagentbench", "longmemeval"]:
            path = prepared / name / "history.jsonl"
            row = first_row(path)
            memory = NativeMemory(output / name / "native", profile=profile)
            write = memory.stage_record(row, scope="integration-case0", ordinal=0)
            recalled = memory.call(
                "recall", {"handle": write["node_id"], "detail_level": "full"}
            )
            expected = json.dumps(
                row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
            if expected not in set(string_values(recalled)):
                raise AssertionError(
                    "exact source history did not round-trip through native recall"
                )
            pulse = memory.milestone(write["node_id"])
            evidence = memory.close()
            memory = None
            report["cases"].append(
                {
                    "campaign": name,
                    "input_sha256": sha(path),
                    "scope": "first prepared source-history record only",
                    "write": write,
                    "recall": recalled,
                    "milestone": pulse,
                    "native": evidence,
                }
            )

        # The official dataset has no typed dependency graph. A supplied small
        # diagnostic trace exercises the real binding without inventing edges.
        fixture_path = Path(__file__).parent / "campaigns/longrca/fixtures/trace.json"
        public, _ = split_record(json.loads(fixture_path.read_text()))
        memory = NativeMemory(output / "longrca" / "native", profile=profile)
        trace = NativeTraceSession(public, TraceBinding(memory))
        trace.ingest()
        selected = public.history[-1]["step"]
        recalled, walked = trace.read_step(selected), trace.causal_walk(selected)
        evidence = memory.close()
        memory = None
        report["cases"].append(
            {
                "campaign": "longrca",
                "scope": "supplied synthetic trace; no role/root prediction",
                "input_sha256": sha(fixture_path),
                "controller_causal_edges_added": 0,
                "read": recalled,
                "walk": walked,
                "diagnostic": trace.diagnostic(),
                "native": evidence,
            }
        )

        path = prepared / "memform" / "runtime-inputs.jsonl"
        packet = first_row(path)
        memory = NativeMemory(output / "memform" / "native", profile=profile)
        committed = commit_packet(memory.stack, memory.session, packet)
        if not committed["handles"] or committed["solver_header_memory_bodies"]:
            raise AssertionError(
                "native ordered writes must return handles, without header injection"
            )
        recalled = memory.call(
            "recall", {"handle": committed["handles"][0], "detail_level": "full"}
        )
        evidence = memory.close()
        memory = None
        report["cases"].append(
            {
                "campaign": "memform",
                "input_sha256": sha(path),
                "scope": "first anonymous prepared memory packet only; no downstream policy score",
                "commit": committed,
                "read": recalled,
                "native": evidence,
            }
        )
        if sources != source_identity():
            raise RuntimeError("source changed during native memory qualification")
        report["status"] = "PASS"
        write_new(output / "proof.json", report)
        return report
    except BaseException as error:
        if memory is not None:
            memory.stack.abort()
        report.update(
            status="FAILED", error_type=type(error).__name__, error=str(error)
        )
        write_new(output / "FAILED.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--profile", choices=["production", "all_rules"], default="all_rules"
    )
    args = parser.parse_args()
    result = qualify(args.prepared, args.output, profile=args.profile)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": len(result["cases"]),
                "model_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
