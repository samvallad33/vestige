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

At the start of a new session, call `session_context` with the current user,
project, and task context. If `session_context` is unavailable or too broad, call
`search` with a concrete query matching the current task.

When accuracy or prior decisions matter, call `deep_reference`. When memories may
conflict, call `contradictions` before answering. Compose retrieved evidence into
the answer; do not merely paste memory summaries.

Save durable preferences, project decisions, recurring corrections, stable facts,
and reusable code patterns with `smart_ingest`. Do not store secrets, credentials,
one-off logs, speculation, or transient command output.

When the user says a memory was useful, call `memory` with `action="promote"`.
When the user says a memory was wrong or unhelpful, call `memory` with
`action="demote"`. When the user explicitly asks to erase a memory permanently,
call `memory` with `action="purge"` and `confirm=true`.
```

## Practical Tool Choices

| Situation | Tool |
|-----------|------|
| Start of session | `session_context` |
| Find exact identifiers, paths, env vars, names | `search` |
| Answer from prior decisions or evolving facts | `deep_reference` |
| Inspect disagreements before answering | `contradictions` |
| Save a preference, decision, correction, or code pattern | `smart_ingest` |
| Retrieve, promote, demote, edit, or purge one memory | `memory` |
| Create a future reminder | `intention` |
| Check health or maintenance state | `system_status` |

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
