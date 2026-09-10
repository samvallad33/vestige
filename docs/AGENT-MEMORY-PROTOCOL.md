# Agent Memory Protocol

> Minimal instructions for any MCP-compatible agent using Vestige.

Vestige is an MCP server, not a Claude-specific workflow. Register `vestige-mcp`
with your client, then give the agent a short instruction that makes memory part
of its normal reasoning loop.

## Register Vestige

Use your client's MCP server configuration format. The command is the same:

```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```

Examples:

```bash
claude mcp add vestige vestige-mcp -s user
codex mcp add vestige -- vestige-mcp
```

## Agent Instruction

Add this to the agent's global or project instruction file:

```text
Use Vestige as durable local memory.

At session start, call `session_start` with queries and current project context.
Use an explicit scope for project memories. For source-aware code context, pass
context.codebase and context.repoPath for the checkout you are actually editing.

Discover the installed toolset with `memory_status` view="tools". Supply
`tool="<name>"` to inspect one tool's complete input schema. This guide is derived
from the server's own tools/list, including every advertised action, mode and
view. Select calls for the task; using every tool in every session wastes context
and can cause unwanted mutations.

Use `recall` mode="lookup" for search (concrete=true for literal identifiers),
mode="reason" for prior decisions, and mode="contradictions" to inspect conflicts.
Check current source evidence before treating memory as fact. Retention is a
retrieval signal, not truth. Graph connections and backfill candidates are
hypotheses; a retrieval receipt records evidence use, not proof of causality.

Save durable preferences, project decisions, recurring corrections, stable facts,
and reusable code patterns with `smart_ingest`. Do not store secrets, credentials,
one-off logs, speculation, or transient command output.

When the user says a memory was useful, call `memory` with `action="promote"`.
When the user says a memory was wrong or unhelpful, call `memory` with
`action="demote"`. When the user explicitly asks to erase a memory permanently,
call `memory` with `action="purge"` and `confirm=true`.
```

## Practical Tool Choices

| Situation | Tool and selection |
|-----------|--------------------|
| Discover all installed tools/actions or inspect exact arguments | `memory_status(view="tools")`; add `tool` for a full schema |
| Start a session with a bounded context packet | `session_start` |
| Search exact identifiers, paths, env vars or names | `recall(mode="lookup", concrete=true)` |
| Reason over earlier decisions or inspect disagreements | `recall(mode="reason")` or `recall(mode="contradictions")` |
| Save durable verified knowledge, singly or in a batch | `smart_ingest` |
| Fetch, inspect state, reinforce, correct or explicitly erase memories | `memory` |
| Remember source-linked patterns/decisions; check or replace reviewed anchors | `codebase` |
| Index and reconcile supported upstream issue systems | `source_sync` |
| Set, check, list, complete, snooze or cancel future intentions | `intention` |
| Inspect health, retention, timeline, audit trail or hygiene | `memory_status` |
| Inspect duplicate candidates, plan/review/apply/undo merges, maintain tags | `dedup` |
| Inspect connections, predictions, recorded compositions and uncombined candidates | `graph` |
| Inspect a retrieval receipt or ablate its frozen evidence | `receipt` |
| Temporarily inhibit a memory or reverse within its supported window | `suppress` |
| Investigate earlier candidates related to a recorded failure | `backfill(promote=false)` first; investigate before promotion |
| Run requested consolidation, dreaming, garbage collection, scoring, backup, export or restore | `maintain` |

Tool annotations apply to the entire tool, including tools mixing read and write
actions. Inspect the specific action's schema and required confirmation. A guide
entry is neither an execution request nor permission. Prefer previews for merge,
tag, garbage-collection and backfill investigations; review before applying.
Never interpret `never_composed` as worldwide novelty or a causal result.

The old `search`, `deep_reference`, `session_context` and `system_status` names
are compatibility redirects. New agents should use the advertised names above.

## What Not To Store

- API keys, tokens, passwords, private keys, or session cookies.
- Raw logs or command output unless the durable lesson is extracted first.
- Guesswork the agent has not verified.
- Temporary plans that will be obsolete after the current session.
- User data the user asked not to retain.

## Portability Notes

The same protocol applies to Claude Code, Codex, Cursor, VS Code, Xcode,
OpenCode, JetBrains, Windsurf, and any other client that can run a stdio MCP server. Claude
Code's Cognitive Sandwich hooks are optional companion files; they are not
required for normal Vestige memory.
