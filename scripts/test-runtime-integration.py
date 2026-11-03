#!/usr/bin/env python3
"""Qualify the installed Python runtime against a Strata MCP server.

Keyword lookup and packet reuse are not Strata operations. The proof is:
force-create, exact handle recall, query recall refused, and provider
request bodies built with zero model calls.
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
            session.discover(["remember", "recall"])
            session.add_user("Remember the synthetic fixture timeout decision")
            created = session.execute_tool(
                "ingest",
                "remember",
                {
                    "content": "RUNTIME_FIXTURE_TIMEOUT uses 25 milliseconds.",
                    "tags": ["runtime-fixture"],
                },
            )
            node_id = created["nodeId"]
            assert node_id.startswith("mem-"), created
            handle = call("recall", {"handle": node_id})
            assert handle.get("isError") is not True, handle
            body = handle.get("structuredContent") or json.loads(handle["content"][0]["text"])
            assert "RUNTIME_FIXTURE_TIMEOUT" in json.dumps(body), body
            refused = call("recall", {"query": "RUNTIME_FIXTURE_TIMEOUT", "mode": "lookup"})
            assert refused.get("isError") is not True, refused
            refused_body = refused.get("structuredContent") or json.loads(refused["content"][0]["text"])
            assert refused_body.get("error") == "handle_required", refused
            requests = {
                provider: session.request(
                    provider,
                    model="synthetic-not-invoked",
                    **({"max_tokens": 100} if provider == "anthropic_messages" else {})
                )
                for provider in ("openai_responses", "anthropic_messages")
            }
            proof = {
                "kind": "installed_runtime_real_stdio_synthetic_fixture",
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "runtime_module": str(Path(vestige_runtime.__file__).resolve()),
                "catalog_count": len(catalog),
                "selected_tool_count": len(requests["openai_responses"]["tools"]),
                "calls": calls,
                "provider_request_bodies": requests,
                "handle_recall_verified": True,
                "query_recall_refused": True,
                "packet_reuse_verified": False,
                "provider_model_calls": 0,
                "local_embeddings": "refused; similarity is not a Strata operation",
                "billing_savings_measured": False,
            }
            if output:
                output.write_text(json.dumps(proof, indent=2) + "\n")
            print(
                "PASS installed runtime, real stdio, handle recall, query refusal; zero provider model calls"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/debug/vestige-mcp"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.binary, args.output)
