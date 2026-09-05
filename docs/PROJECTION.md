# Markdown projection

Vestige stays the source of truth. `project` renders the durable subset of a scope into the rule files other agent clients already read, inside a fenced region the store owns. Everything outside the fence is the human's and is never touched.

## What gets projected

- `decision` and `pattern` memories.
- `fact` and `note` memories tagged `rule`, `preference` or `convention`.
- Only memories that are currently valid (nothing superseded or expired), at or above a retention floor (default 0.3), in the requested scope (default `user`), newest first within each group, capped at `max_items` (default 60).

Each projected line ends with the id of the memory it came from:

```
- Release from an integration branch, never from a feature branch <!-- vestige:2f1c... -->
```

A rule can be traced back to its evidence, and a later re-import can tell an edited line from a generated one.

## Formats

| format | shape | typical target |
|---|---|---|
| `claude-md` | a `## Vestige memory (projected)` section with Decisions, Patterns, Rules and preferences | `CLAUDE.md`, `AGENTS.md` |
| `memory-md` | one line per memory, `[type]` prefixed | `MEMORY.md`, an index file |

## The fence

```
<!-- vestige:projection:begin scope=user format=claude-md -->
...
<!-- vestige:projection:end -->
```

Writing replaces exactly this region. A file without a fence gets one appended. The rendering carries no timestamps, so an unchanged store projects to an unchanged file and a second write is a no-op.

## MCP

```json
{ "name": "project", "arguments": { "path": "CLAUDE.md" } }
```

Preview is the default: it returns the region, the items with their ids, and a line diff against the target. Writing needs `"action": "write"` and `"confirm": true`. The target must stay inside `root` (the server's working directory unless given); a path that escapes it is refused.

## CLI

```
vestige project --out CLAUDE.md            # print the diff
vestige project --out CLAUDE.md --write    # apply it
vestige project --format memory-md --out MEMORY.md --scope my-project --write
```

## Not yet

Re-importing human edits made inside the fence back into memory (through merge or supersede review) is the second half of the roadmap item and is tracked separately. Until then, edit memories in Vestige and re-project.
