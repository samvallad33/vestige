# Codex Protocol for Vestige

> Written for Vestige 4.x.

Codex can connect to Vestige through MCP, but MCP registration alone only makes the tools available. It does not make Codex call them. Use this protocol when you configure a Codex workspace that should consult Vestige before it answers.

Vestige finds a record by an exact handle: a full id, a unique id prefix of 8 or more characters, or an exact tag. It does not search by wording, so the protocol below names the tag to recall.

## 1. Register Vestige MCP

```toml
[mcp_servers.vestige]
command = "/absolute/path/to/vestige-mcp"
```

Restart Codex after changing MCP configuration.

## 2. Add an `AGENTS.md` trigger

Codex reads `AGENTS.md` files as workspace instructions. Put a file at the repo root, or a higher workspace root, with a rule like:

```markdown
At the start of a task, call Vestige `session_start` with include_intentions and
include_status, and with context.codebase and context.repoPath for the checkout you
are editing. Then call `recall` with handle set to the task's narrow topic tag.
Recall by exact tag or id only; a free-text query returns similarity_disabled.
Compose what you find into actions; do not summarize retrievals. Save decisions,
corrections and verified facts with `smart_ingest`, tagged with the project and one
narrow topic.
```

## 3. Use a router

Use the smallest call that can change the answer:

- `session_start`: start of a topic or project switch.
- `recall` with `handle`: the history of one topic by exact tag, or one record by id.
- `memory` with `action="get_batch"`: expand specific load-bearing records by id.
- `causal_walk` and `forgotten_lesson`: after a failure. Save the failure as an `event` record first and walk from its id.
- `smart_ingest`: save durable corrections, decisions and new preferences.

## 4. Compose, do not summarize

A retrieved record is evidence, not the final answer.

Use this mental transform:

```text
record -> implication -> action
```

If a record does not change the action, do not mention it. If it does, make the changed recommendation clear and cite the record id.

## 5. Know the limit

Claude Code's Cognitive Sandwich uses `UserPromptSubmit` and `Stop` hooks, and most of its layers call v3 tools. See [COGNITIVE_SANDWICH.md](../COGNITIVE_SANDWICH.md). This repository ships no Codex hook adapter.

For Codex, the portable layer is:

1. MCP server configured.
2. `AGENTS.md` instruction trigger.
3. Explicit agent discipline: call Vestige at the start of a task and save what the task established.

To stop destructive commands before they run, use the [Operator Lite](../../operator-lite/README.md) gate. It supports Codex hooks and is a separate layer from this protocol.
