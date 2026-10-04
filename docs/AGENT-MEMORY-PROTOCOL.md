# Agent Protocol

> Written for Vestige 4.x.

> Minimal instructions for any MCP-compatible agent using Vestige.

Vestige is an MCP server, not a Claude-specific workflow. Register `vestige-mcp` with your client, then give the agent a short instruction that makes Vestige part of its normal loop.

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
Use Vestige as the durable log of decisions, corrections, facts and failures on this machine.

At session start, call `session_start` with include_intentions and include_status.
In a repository, also pass context.codebase and context.repoPath for the checkout
you are editing. Then call `recall` with handle set to the task's narrow topic tag.
`session_start` ignores `queries` and says so in its notices.

Vestige finds a record only by an exact handle: a full id, a unique id prefix of 8
or more characters, or an exact tag. Tags are case-sensitive. A free-text query
returns similarity_disabled. A tag recall returns every record under that tag, from
every scope, so recall narrow tags only.

Discover the installed tools with `memory_status` view="tools". Supply tool="<name>"
to inspect one tool's complete input schema. Call only what the task needs; using
every tool in every session wastes context and can cause unwanted writes.

A record is context, not proof. Check current source evidence before you treat it as
fact. A link counts only if the log recorded it: a typed edge, a woven outcome or a
receipt. Retention is a scheduling signal, not truth.

Save durable preferences, project decisions, corrections, verified facts and reusable
code patterns with `smart_ingest`. Tag every save: one project tag and one or two
narrow topic tags, spelled the same way every time. Pass `links` to declare a cause.
Do not store secrets, credentials, one-off logs, speculation or transient command output.

When the user says a record was useful, call `memory` with action="promote". When the
user says it was wrong, call `memory` with action="demote", then action="edit" to
admit the corrected version. An edit changes the id.

Nothing is erased. `suppress` hides a record from every read and cannot be undone.
`purge` and `memory` actions purge and delete return unavailable_in_4_0.
```

## Practical Tool Choices

| Situation | Tool and selection |
|-----------|--------------------|
| Discover all installed tools and actions, or inspect exact arguments | `memory_status(view="tools")`; add `tool` for a full schema |
| Start a session with a bounded context packet | `session_start` |
| Load the history of a topic, or fetch one record | `recall(handle=...)` with an exact tag or an id |
| Save durable knowledge, singly or up to 20 at a time, with typed `links` | `smart_ingest` |
| Fetch, inspect state, promote, demote or correct records | `memory` |
| Remember source-linked patterns and decisions; check or replace anchors; turn commits into change records | `codebase` |
| Set, check, list, complete, snooze or cancel future intentions; evaluate plans against evidence | `intention` |
| Inspect health, retention, timeline, audit trail, provenance, coverage and counts | `memory_status` |
| List exact duplicates, undo a recorded write, preview a tag rename or merge | `dedup` |
| Propose never-composed pairs with proofs, weave what a tested pair showed, inspect woven compositions | `ghostlink` |
| Walk a failure backward over recorded edges | `causal_walk` from a record that holds the symptom, with `node_id` on each start point |
| Find faded fix or lesson records behind a failure | `forgotten_lesson` |
| See what a write did, or replay it against the log | `receipt` |
| Check that the walk finds a planted cause, on a throwaway copy | `selftest` |
| Write the durable subset of a scope into a fenced region of `CLAUDE.md` or `MEMORY.md` | `project` (preview first) |
| Hide a record from every read | `suppress` (cannot be undone on Strata) |
| Replay recorded edges (`dream`, `dream_compile`), back up or export | `maintain`. `gc` reports no candidates and deletes nothing on a Strata log |

Tool annotations apply to the whole tool, including tools that mix read and write actions. Inspect the specific action's schema and its confirmation field. A guide entry is neither an execution request nor permission. Prefer previews, and review before you apply.

A `ghostlink` proposal is a pair no recorded edge has composed. It is neither proof of causality nor of worldwide novelty. Test the pair, then weave the outcome.

Tool names from v3, such as `search`, `deep_reference`, `explore_connections` and `memory_graph`, are not advertised in 4.x. Use the names above. `source_sync` is absent from a default build.

## What Not To Store

- API keys, tokens, passwords, private keys, or session cookies. The gate refuses the known credential shapes, and it does not catch a bare `password: ...` line.
- Raw logs or command output, unless the durable lesson is extracted first.
- Guesswork the agent has not verified.
- Temporary plans that will be obsolete after the current session.
- User data the user asked not to retain.

## Portability Notes

The same protocol applies to Claude Code, Codex, Cursor, VS Code, Xcode, OpenCode, JetBrains, Windsurf, and any other client that can run a stdio MCP server. Claude Code's Cognitive Sandwich hooks are optional companion files; they are not required to use Vestige.
