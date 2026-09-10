# Vestige runtime integration (v3 alpha)

Install the local package in your application's environment:

```sh
python -m pip install ./integrations/python
```

The package owns an explicit transcript, progressively selected native tool
schemas, and retained recall packets. It starts no model agent and creates no
SDK client. Your application supplies its own Vestige command and SDK calls.

```python
from vestige_runtime import DeveloperSession, StdioMcp

with StdioMcp(["vestige-mcp", "--no-http", "--data-dir", "./local-memory"]) as mcp:
    session = DeveloperSession(mcp.catalog(), mcp.call)
    session.discover(["recall"])
    session.add_user("Find the project's timeout decision")
    result = session.execute_tool("lookup-1", "recall", {"query": "timeout"})
    body = session.request("openai_responses", model="YOUR_MODEL")
    # Pass body to your own SDK or benchmarks/task-cost/capture.py.
    # Anthropic uses provider="anthropic_messages", with max_tokens supplied.
```

`execute_tool` is explicit execution authorized by the host; this package does
not implement a model loop or mutation approval policy. The host must decide
which requested calls to execute. Discovery returns exact native schemas and
keeps only selected tools in serialized requests.

Repeated lookup acknowledges a complete packet only while its full evidence
remains in this session's transcript. After compaction, checkpoint restoration,
or uncertain retention, call `reset_context(summary=...)` and recall again.
Incomplete packets cannot be acknowledged. Context overflow raises an error;
the application must explicitly compact and refresh. Do not mutate `events` or
send only fragments of the generated request while relying on its cache state.

OpenAI Responses and Anthropic Messages request bodies are serialized locally;
live SDK/provider qualification and provider caching remain separate checks.
Smaller bodies are not measured token-fee savings. No automatic installation
into Codex, Claude, or another host occurs.

The synchronous stdio transport is tested on POSIX, with argv execution (no
shell), bounded responses, deadlines and an owned child process. It uses only a
small base environment plus explicit `env` overrides, and discards server stderr.
Use a trusted local server binary. Windows transport is not qualified.

Run `python -m unittest discover -s integrations/python/tests -v`. For a real
server check, run `scripts/test-runtime-integration.py --binary PATH` against a
fresh disposable store; no model API is invoked.
