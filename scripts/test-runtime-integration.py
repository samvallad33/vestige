#!/usr/bin/env python3
"""Qualify the installed Python runtime against a disposable real MCP server.

Requires vestige-runtime installed in the invoking Python environment. No SDK or
model is called. The transcript contains only synthetic fixture information.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import vestige_runtime
from vestige_runtime import DeveloperSession, StdioMcp


def run(binary, output):
    calls = []
    with tempfile.TemporaryDirectory(prefix="vestige-runtime-proof-") as temp:
        with StdioMcp(
            [str(binary.resolve()), "--no-http", "--data-dir", temp],
            env={"VESTIGE_DASHBOARD_ENABLED": "false", "RUST_LOG": "error"},
            timeout=60,
        ) as mcp:
            catalog = mcp.catalog()

            def call(name, arguments):
                result = mcp.call(name, arguments)
                calls.append({"tool": name, "arguments": arguments, "result": result})
                return result

            session = DeveloperSession(catalog, call)
            session.discover(["smart_ingest", "recall"])
            session.add_user("Remember the synthetic fixture timeout decision")
            session.execute_tool(
                "ingest",
                "smart_ingest",
                {
                    "content": "RUNTIME_FIXTURE_TIMEOUT uses 25 milliseconds.",
                    "forceCreate": True,
                },
            )
            args = {"query": "RUNTIME_FIXTURE_TIMEOUT", "mode": "lookup"}
            first = session.execute_tool("lookup-1", "recall", args)
            assert first["notModified"] is False and first.get("packetId"), first
            assert first["results"], first
            second = session.execute_tool("lookup-2", "recall", args)
            assert second["notModified"] is True and not second.get("results"), second
            requests = {
                provider: session.request(
                    provider,
                    model="synthetic-not-invoked",
                    **({"max_tokens": 100} if provider == "anthropic_messages" else {})
                )
                for provider in ("openai_responses", "anthropic_messages")
            }
            session.reset_context(summary="Continue the synthetic timeout task")
            third = session.execute_tool("lookup-3", "recall", args)
            assert third["notModified"] is False and third["results"], third
            assert "known_packet_id" not in calls[-1]["arguments"]
            proof = {
                "kind": "installed_runtime_real_stdio_synthetic_fixture",
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "runtime_module": str(Path(vestige_runtime.__file__).resolve()),
                "catalog_count": len(catalog),
                "selected_tool_count": len(requests["openai_responses"]["tools"]),
                "calls": calls,
                "provider_request_bodies": requests,
                "packet_reuse_verified": True,
                "compaction_refresh_verified": True,
                "provider_model_calls": 0,
                "local_embeddings": "server configured runtime may run",
                "billing_savings_measured": False,
            }
            if output:
                output.write_text(json.dumps(proof, indent=2) + "\n")
            print(
                "PASS installed runtime, real stdio, retained packet reuse and compaction refresh; zero provider model calls"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.binary, args.output)
